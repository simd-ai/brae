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
// And RAS kOmegaSST, host and `-device`, in the ordinary lineage: RAS/waterChannel, gated in
// tests/interfoam_waterchannel_vs_openfoam.sh, and its ASSEMBLED SYSTEM against OpenFOAM's own in
// tests/interfoam_sst_assembly_vs_openfoam.sh. Under `density variable` the SST is refused on BOTH arms
// (the case reader's; no shipped tutorial pairs the two -- all five SST cases run the default density),
// and the kEpsilon pair runs there WITH the mangrove turbulence option --
// -Sp(rho*coeff) rather than -Sp(coeff) -- gated by `densityVariable` in
// tests/interfoam_mangrove_vs_openfoam.sh.
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
// a processorAgglomerator, and any smoother but DIC, DICGaussSeidel, GaussSeidel and symGaussSeidel --
// the device's GAMG runs those same four. Any OTHER solver for p_rgh still substitutes, under a notice.
//
// AND THE SMOOTHER REFUSALS SAY WHICH KIND THEY ARE, because two of them are not gaps. OpenFOAM selects a
// smoother from the SYMMETRIC table when the matrix is symmetric and the ASYMMETRIC one otherwise
// (lduMatrixSmoother.C:29-57), and the registrations split: DIC/DICGaussSeidel/FDIC are symmetric-only,
// DILU/DILUGaussSeidel asymmetric-only, the Gauss-Seidels both. brae's gamgSolve refuses an asymmetric
// matrix outright, so `smoother DILU` on p_rgh is invalid IN OPENFOAM TOO -- measured on
// laminar/waves/stokesI, where real OpenFOAM stops with "Unknown symmetric matrix smoother type DILU".
// Porting it would accept a case OpenFOAM rejects, so neither path ever should; FDIC and
// nonBlockingGaussSeidel are the two that really are missing work, and the refusal names the difference.
//
// AND A MOVING MESH, on the host: dynamicMotionSolverFvMesh with the solidBody solver moving the
// whole mesh under any of OpenFOAM's motion functions but drivenLinearMotion
// (src/dynamicFvMesh, src/meshTools/solidBodyMotionFunctions), or with displacementLaplacian DEFORMING
// it from waveMaker paddles (src/fvMotionSolver, src/waveModels/derivedPointPatchFields), with
// movingWallVelocity walls, the relative flux, Uf and the old volumes where interFoam.C and pEqn.H
// read them; CorrectPhi after every update under `correctPhi` (the default on a moving mesh) and at
// the start of EVERY case (initCorrectPhi.H); a wave absorber on a moving mesh; and a CLOSED tank's
// pressure reference (pRefCell or pRefPoint, pRefValue, adjustPhi). `-device` RUNS the moving mesh too
// (device_moving, device_moving_gamg, device_moving_SST) and the closed tank, adjustPhi included on
// both arms -- stopping where OpenFOAM stops (closed_abort_host, closed_abort_device) -- and a permeable
// wall on a moving mesh (device_permeable_moving). Still refused by name: a cellZone or cellSet, every other motionSolver, a
// turbulent case on a moving mesh, points0, and a restart of a moved mesh. testTubeMixer, the five
// sloshing tanks and the five waveMakers run -- waveMakerPiston and waveMakerFlap gated with their
// pressure solves converged, and the two multi-paddle ones approximating p_rgh under a notice, since
// their `p_rgh { $pcorr; }` names a pattern-keyed entry brae's dictionary expansion does not resolve.
//
// AND AN ADAPTIVE MESH, on the host: dynamicRefineFvMesh refining and unrefining on the case's own
// driving field, through hexRef8 and removeFaces, with every field the change invalidates mapped --
// alpha1, U and p_rgh with their patch fields, phi, rhoPhi, nHatf, Uf, rAU, alpha2's patch values and
// every old-time level -- and the solver's own rebuild after it: gh, ghf, the mixture, the curvature,
// `phi = Sf & Uf` and the pcorr solve that makes the mapped flux divergence-free again.
// tests/interfoam_amr_vs_openfoam.sh holds laminar/damBreakWithObstacle against OpenFOAM's own two
// steps: the same mesh cell for cell, alpha 1.9e-15, U 2.4e-13 relative, and all nine pressure solves
// on OpenFOAM's iteration count and residual. Still refused BY NAME: a moving mesh beside refinement,
// turbulence, waves, MRF, fvOptions, CrankNicolson, a pressure reference, a CN restart directory,
// `correctPhi no`, a `correctFluxes` entry naming a velocity or NaN, and a coupled patch on a refining
// mesh. `-device` RUNS IT TOO: the change is host work on either loop -- topology surgery and six
// integer maps -- and the device branch owes the round trip, every mesh-sized buffer down to the host
// and back up onto a DeviceMesh rebuilt from scratch, which is what invalidates every schedule cache
// keyed on its addressingId. Gated in the same test: alpha 2.2e-15 from OpenFOAM, p_rgh 6.1e-15
// relative, U 3.2e-13, and the arm run TWICE in one process bit-identical (the detector for a cache
// keyed on a recycled pointer). Both arms refuse refinement beside MOTION (device_refine_motion). A 2-D
// case RUNS: its hull average put an `empty` patch's stored values where OpenFOAM has zeros -- a refined
// 2-D mesh has internal faces in the empty direction -- which read alpha 5.2e-03 from OpenFOAM and reads
// 1.8e-15 with FluxMeshView::patchHoldsNoValues carrying the distinction.
//
// AND `Gauss interfaceCompression` on the alpha fluxes, on BOTH paths: the PhiScheme four waveMakers
// name for div(phirb,alpha) -- two cell values and no gradient, so it carries onto the device whole.
//
// AND THE CASE'S NON-ORTHOGONAL CORRECTIONS, on the host: `corrected` and `limited` laplacians and
// snGrads on a mesh that is not orthogonal (the tanks' 44 degrees), through the pressure equation,
// the viscous term and the three snGrads -- AND `uncorrected` and `limited 0`, which take those same
// nonOrthDeltaCoeffs with the correction flux left off (uncorrectedSnGrad.H:113-119). Both used to run
// ORTHOGONAL: `uncorrected` behind a refusal, `limited 0` behind nothing at all. `-device` RUNS all of it
// (device_sheared_corrected, device_sheared_uncorrected).
//
// AND EVERY gradSchemes ENTRY by the name its call site asks for -- `Gauss linear`, `leastSquares` and
// `cellLimited` over either -- on the host (laminar/damBreak `gradLsqLimited`, `nHatLimited`) and on the
// device, whose scalar gradients and grad(U)'s dev2 term take them: RAS/electrostaticDeposition
// (`default cellLimited leastSquares 1`) runs on both arms (tests/interfoam_moving_vs_openfoam.sh
// `esd`). What the device still refuses is a leastSquares grad(U) beside a site its momentum assembly
// builds Gauss: a deferred correction, a V-scheme limiter or a corrected laplacian (device_gradLsq).
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
// `-device` RUNS the pair too -- validation/interFoamCyclic holds both arms against OpenFOAM
// (tests/interfoam_cyclic_vs_openfoam.sh), and device_baffle/device_baffle_SST/device_les_cyclic hold
// the baffle. What is refused across a pair is refused on both paths, by the list above.
// A cyclicAMI PAIR runs on `-device` as well, on a static mesh and on one that MOVES: the pair's weighted
// stencil rides beside its faces (DeviceCyclic's CyclicNbr) and every kernel reads the neighbour the
// host's patchNeighbourValue does; after a move the host stage recomputes the AMI and the device takes
// the pair again in place (refreshDeviceCyclicAfterMove), with the mesh flux and the absolute flux on
// its faces. tests/interfoam_ami_device/ holds RAS/mixerVesselAMI against OpenFOAM, static and rotating,
// nine controls beside it, and the write gate holds the tutorial as shipped on both arms. Still refused
// on the device across a moving mesh: a plain cyclic and a cyclicACMI.
//
// AND LES kEqn, on the host: the uniform lineage, with the cubeRootVol or smooth filter width, on an
// axisymmetric wedge -- whose host matrix coefficients and gradient patch value it took to get there.
// tests/interfoam_les_vs_openfoam.sh holds LES/nozzleFlow2D against OpenFOAM, on BOTH arms (device_les)
// -- the wedge took three device defects of its own to get there.
//
// AND the CrankNicolson ddt scheme, on BOTH loops -- the momentum equation, ddtCorr and the kEpsilon
// closure under it, alphaEqn.H's own off-centred flux and end-of-step un-blend, on a mesh that does not
// move: RAS/damBreak with `default CrankNicolson 0.5`, held by tests/interfoam_cn_vs_openfoam.sh. The one
// shipped tutorial that names the scheme, RAS/floatingObject, also moves its mesh under rigidBodyMotion,
// which the host loop carries: it runs, and is refused at its first write time for CrankNicolson's
// old-time fields (census 2026-09-30).
//
// AND the mangrove fvOptions, on BOTH loops -- multiphaseMangrovesSource on U, and
// multiphaseMangrovesTurbulenceModel on k and epsilon under kEpsilon, whose k and epsilon may be solved
// with PBiCG and DILU: waves/mangroveInteraction, held by tests/interfoam_mangrove_vs_openfoam.sh.
//
// AND a coincident cyclicACMI pair, on the host -- a createBaffles baffle whose `scale` (a constant, a
// table or a coded per-face PatchFunction1) opens and shuts faces with time: RAS/damBreakLeakage, held by
// tests/interfoam_leakage_vs_openfoam.sh, on both arms (device_leak). Its EXPLICIT MULES branch is
// refused on BOTH arms by the same rule (leak_explicitMULES, leak_explicitMULES_device): the tutorial ships
// `MULESCorr yes`, and the rescale point is gated there only.
//
// AND `Gauss limitedLinear` on div(rhoPhi,U), on the host: one magSqr limiter per face, as OpenFOAM's
// vector form has it. tests/interfoam_limitedlinear_vs_openfoam.sh holds eulerianInjection against
// OpenFOAM, on both arms (device_limitedLinear) -- the device branch accumulated magSqr(U) into a pool
// block resize() had not zeroed until that arm was lifted.
//
// WHAT IT WILL NOT RUN. Every refusal the components carry is in force: every LESModel but kEqn, every
// RASModel but kEpsilon and kOmegaSST, any ddtSchemes default but Euler and CrankNicolson (the second
// on both loops now -- tests/interfoam_cn_vs_openfoam.sh, device_cn), every fvOption but
// explicitPorositySource/DarcyForchheimer (host only: tests/interfoam_angledduct_vs_openfoam.sh) and
// the two mangrove sources (both loops), MRF beside a moving mesh, RAS or a
// fixedFluxPressure patch -- BOTH loops run it otherwise, all four of interFoam's MRF calls
// (tests/interfoam_mrf_vs_openfoam.sh holds laminar/mixerVessel2D on the host arm and the device arm
// at the same bounds) -- and a case that
// omits nAlphaCorr, nAlphaSubCycles, cAlpha, maxAlphaCo or -- under MULESCorr -- nLimiterIter. Each
// throws by name. (maxAlphaCo at a FIXED time step too, where OpenFOAM reads it as well; not under
// localEuler, where it does not -- tests/interfoam_write/refusal/max_alpha_co.sh.) And PIMPLE's
// `residualControl` beyond one outer corrector, where OpenFOAM would leave the outer correctors early
// (tests/interfoam_write/refusal/pimple_residual_control.sh).
//
// THAT LIST WAS NOT TRUE when it was written: MRF and fvOptions were named here and refused nowhere,
// and a moving or refining mesh was not even named. brae was run over all 44 shipped tutorials and the
// ones that reached `End:` were counted -- two did that should not have, both on `dynamicRefineFvMesh`.
// Also refused now: any `dynamicFvMesh` but staticFvMesh and the solid-body motion above, a dictionary-form `sigma` and a
// missing one (both used to become ZERO surface tension), and a setTimeStep function object. The
// device loop runs nOuterCorrectors above 1 and nNonOrthogonalCorrectors above 0 now (device_nOuter2,
// device_nNonOrth1); it used to refuse both.
// tests/interfoam_refusals.sh holds every one of them, each beside the form OpenFOAM treats as nothing
// -- `staticFvMesh`, `active no` -- which must still RUN.
//
// WHAT `-device` REFUSES, by the arm that holds each one. This list is CHECKED: interfoam_refusals.sh
// compares it against its own `device_*` arms and fails if the two disagree, so a refusal that is
// lifted or added without editing this header stops the gate. Nine of the sentences above said
// `-device` refuses something it had run for units -- LES, a cyclic, a moving mesh, kOmegaSST,
// interfaceCompression, the leakage pair, limitedLinear, the non-orthogonal corrections,
// nOuterCorrectors above 1 -- which is why the list is machine-checked rather than described.
//
// BEGIN DEVICE REFUSALS
//   device_gamg_smootherDILU device_gradLsq
//   device_refine_motion device_eulerU_cnAlpha_moving device_eulerU_cnAlpha_refining
//   device_prevCorr_pair device_alpha2Patches_refining device_smoothCurvature
//   device_gradUCache device_gradUCacheKEpsilon device_gradUCacheLimited device_gradUCacheCoupled
// END DEVICE REFUSALS
#include "device_schedule.cuh"
#include "host_allocator.cuh"
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "inter_driver_cpp.cuh"
#include "start_time.cuh"
#include "time_instances.cuh"
#include "foam_dict.cuh"
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    // before the first large allocation: see host_allocator.cuh
    const std::string hostAllocatorNotice = brae::setHostAllocator();
    // before any CUDA call: see device_schedule.cuh
    brae::setCudaSchedule();
    bool onDevice = false;
    std::string caseDir = ".";
    // WHAT `brae` HANDS ON IS WHAT THIS TAKES: the launcher forwards its own arguments unchanged
    // (solver_dispatch.cuh), and a user may have typed the case as `-case <dir>` or as a bare directory. A bare
    // directory was not read here -- `brae myCase` ran the current directory -- and an option this binary does
    // not have was skipped without a word; it is refused by name below.
    std::string unknownOption;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-case" && i + 1 < argc)
        {
            caseDir = argv[++i];
        }
        else if (a == "-device")
        {
            onDevice = true;
        }
        else if (a == "-help" || a == "--help" || a == "-h")
        {
            std::printf("brae_interFoam: OpenFOAM's interFoam, re-ported.\n"
                        "  usage: brae_interFoam [-case] <dir> [-device]\n"
                        "    -device  run the time loop on the GPU (runInterFoamDevice). The same\n"
                        "             components and the same case translation; the boundary\n"
                        "             conditions stay on the host. Without it the HOST loop runs: the\n"
                        "             reference the gates hold to OpenFOAM, on one core.\n"
                        "  `brae <dir>` on a case whose controlDict says `application interFoam` runs\n"
                        "  this binary with -device.\n");
            return 0;
        }
        else if (!a.empty() && a[0] != '-')
        {
            caseDir = a;
        }
        else if (unknownOption.empty())
        {
            unknownOption = a;
        }
    }

    try
    {
        using namespace brae;
        using namespace brae::cpu::interFoam;

        if (!unknownOption.empty())
        {
            throw std::runtime_error(
                "the option `" + unknownOption + "` is not one this solver takes. It takes `-case <dir>` or a "
                "bare case directory, and `-device` for the GPU loop.");
        }
        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        // THE ENTRIES Time READS, AS Time READS THEM (Time.C:146-190, TimeIO.C:268-356). `deltaT` is
        // mandatory -- it defaulted to 1e-3 here; `startFrom` defaults to latestTime -- it defaulted to
        // startTime, so a case without the entry restarted from 0 over its own output; any other word than
        // the three stops OpenFOAM -- it ran as startTime; `startTime` is mandatory under `startFrom
        // startTime`; and `stopAt` other than endTime ends the run at a write or at once (Time.C:1163-1181)
        // -- it was not read at all. That last one is refused: the loops stop at endTime alone.
        if (!controlDict.found("deltaT"))
        {
            throw std::runtime_error(
                "brae interFoam: system/controlDict has no `deltaT`. Time reads it with no default "
                "(TimeIO.C:272-275) and OpenFOAM stops.");
        }
        const std::string stopAt = controlDict.wordOr("stopAt", "endTime");
        if (stopAt != "endTime")
        {
            throw std::runtime_error(
                "brae interFoam: system/controlDict sets `stopAt " + stopAt + "`. Time then ends the run at "
                "the next write, or at once (TimeIO.C:340-356, Time.C:1163-1181); the loops here stop at "
                "endTime alone. Only `stopAt endTime` is ported.");
        }
        if (controlDict.found("stopAt") && !controlDict.found("endTime"))
        {
            throw std::runtime_error(
                "brae interFoam: system/controlDict sets `stopAt endTime` and no `endTime`; Time reads it with "
                "no default there (TimeIO.C:344-347) and OpenFOAM stops.");
        }
        const scalar endTime = controlDict.scalarOr("endTime", scalar(0));
        const scalar deltaT0 = controlDict.scalarOr("deltaT", scalar(0));
        const std::string startFrom = controlDict.wordOr("startFrom", "latestTime");
        if (startFrom != "startTime" && startFrom != "firstTime" && startFrom != "latestTime")
        {
            throw std::runtime_error(
                "brae interFoam: system/controlDict sets `startFrom " + startFrom + "`; expected startTime, "
                "firstTime or latestTime (Time.C:184-190).");
        }
        if (startFrom == "startTime" && !controlDict.found("startTime"))
        {
            throw std::runtime_error(
                "brae interFoam: system/controlDict sets `startFrom startTime` and no `startTime`; Time reads "
                "it with no default there (Time.C:155-158) and OpenFOAM stops.");
        }
        // `startFrom latestTime` IS HONOURED, and it used to be read and then thrown away: anything other
        // than `startTime` fell to 0, so the standard way to CONTINUE a run silently restarted it from the
        // beginning. resolveStartTime is the resolution the other drivers already shared; the probe is
        // alpha.water rather than U, because interFoam's own restart is a phase field and a mesh-only
        // directory written by snappyHexMesh carries neither.
        char startBuf[64];
        std::snprintf(startBuf, sizeof startBuf, "%g",
                      (double)controlDict.scalarOr("startTime", scalar(0)));
        // the probe is the case's OWN phase field: `alpha.water` here sent a `startFrom latestTime` restart
        // of any other phase name (hifoam's alpha.liquidN2) silently back to startTime, where the writer
        // would then overwrite and purge the run's own output
        const std::string phase1 = cpu::twoPhase::readTransportProperties(caseDir).phase1Name;
        const std::string alphaProbe = "alpha." + phase1;
        const std::string startName =
            resolveStartTime(caseDir, startFrom, startBuf, alphaProbe.c_str());
        const scalar startTime = static_cast<scalar>(std::strtod(startName.c_str(), nullptr));

        // WHICH TIME DIRECTORY THE MESH COMES FROM, which is a search and not a path: a run that refined or
        // moved its mesh wrote a newer one into a time directory, and polyMesh resolves points, faces and
        // boundary separately (cpu::timePaths::meshInstances, mirroring polyMesh.C:175-245). This used to be
        // `constant/polyMesh` in four places, so a genuine restart read the mesh the case STARTED from.
        const cpu::timePaths::MeshInstances mi =
            cpu::timePaths::meshInstancesForStartDir(caseDir, caseDir + "/" + startName);
        if (mi.points != "constant" || mi.faces != "constant")
        {
            std::fprintf(stderr, "brae: mesh instances -- points '%s', faces '%s', boundary '%s'\n",
                         mi.points.c_str(), mi.faces.c_str(), mi.boundary.c_str());
        }
        PrimitiveMesh m;
        m.read(mi.pointsDir(caseDir), mi.facesDir(caseDir), mi.boundaryDir(caseDir));
        FvGeometry g;
        g.build(m);
        // mirrorACMI: this loop couples a coincident cyclicACMI pair itself; the device loop, handed the
        // same patches uncoupled, refuses it by name
        std::vector<FvPatch> patches = buildPatches(m, g, /*mirrorACMI=*/true);
        cpu::cyclicACMI::Interfaces acmi;
        cpu::cyclicAMIFvPatch::Interfaces amiPairs;
        // A COINCIDENT cyclicACMI PAIR IS COUPLED FIRST, ON BOTH PATHS: its masks split the face
        // areas, which moves the cell geometry every patch is built from, and setup() REBUILDS
        // `patches` (cyclic_acmi_cpp.cu) -- so a plain cyclic attached before it would lose its
        // coupling. That was the order here, the reverse of the contract cyclic_acmi_cpp.cuh states
        // and of the gate's harness; it was loud rather than silent (the case-build refusal named the
        // uncoupled cyclic) and no shipped case carries both.
        acmi = cpu::cyclicACMI::setup(m, g, patches, startTime);
        // A PLAIN CYCLIC IS COUPLED ON BOTH PATHS. It used to be host-only, because the device loop
        // branched on nothing and a coupled patch would have been run as a wall; the device loop now
        // carries the pair -- its matrices, its fluxes, MULES and nHatf, each gated on its own -- so
        // withholding the coupling here would refuse a case the loop can run.
        attachCyclicCoupling(patches, m, g);
        // A cyclicAMI IS COUPLED ON BOTH PATHS: the device loop carries the pair's weighted stencil
        // beside its faces (DeviceCyclic's CyclicNbr) and reads the neighbour the host does.
        // the BOUNDARY instance: the AMI entries are read from polyMesh/boundary, which polyMesh
        // resolves separately from the faces and never older than them
        amiPairs = cpu::cyclicAMIFvPatch::setup(mi.boundaryDir(caseDir), m, g, patches);
        // ...handed to the host loop mutable as well, for a case whose mesh moves (MutableMesh)
        MutableMesh mutableMesh;
        mutableMesh.m = &m;
        mutableMesh.g = &g;
        mutableMesh.patches = &patches;
        mutableMesh.acmi = &acmi;
        mutableMesh.ami = &amiPairs;

        // The start directory OpenFOAM would use. `0` is written as `0` and not `0.000000`, which is what
        // every tutorial ships -- and under `startFrom latestTime` it is the directory resolveStartTime
        // found, whose NAME is what OpenFOAM's timeName() would be (so `0.002` and not `0.0020000`).
        const std::string startDir = caseDir + "/" + startName;

        // An upper bound on the steps, and endTime below is the REAL bound. A fixed-step case needs
        // exactly (end-start)/dt; an adjustTimeStep case cannot be counted ahead of time, because
        // deltaT is chosen from a Courant number that does not exist yet. This count used to be the
        // only bound and the comment here claimed otherwise: damBreak at `endTime 0.004` ran its 40
        // steps out to t = 0.054, thirteen times past the end of the case. The 4x allows the step to
        // grow at setDeltaT's 1.2 cap for eight steps before endTime has to stop the loop.
        // UNDER adjustTimeStep endTime is the only bound: a Courant-limited run whose step averages
        // below deltaT/4 -- a filling case -- stopped at the old 4x count before endTime, printed `End:`
        // and said nothing. The count stays the bound of a fixed-step case.
        // the reader the time loop's own controls use (readTimeControls.H), not a second parse of the entry
        const bool adjustTimeStep = TimeControls::read(controlDict).adjustTimeStep;
        const label nSteps = (deltaT0 > scalar(0))
            ? (adjustTimeStep ? label(2000000000)
                              : static_cast<label>(scalar(4)*(endTime - startTime) / deltaT0 + scalar(0.5)))
            : label(0);
        if (nSteps < 1)
            throw std::runtime_error(
                "brae interFoam: controlDict gives no steps to take (endTime " +
                std::to_string((double)endTime) + ", deltaT " + std::to_string((double)deltaT0) + ").");

        std::printf("brae interFoam (OF-mirror): %ld cells, start %s, endTime %g\n",
                    (long)m.nCells(), startName.c_str(), (double)endTime);
        std::printf("%s", hostAllocatorNotice.c_str());

        // the time directories, at OpenFOAM's write times (inter_writer_cpp.cuh)
        cpu::interFoam::InterWriter writer(caseDir, startDir, patches, phase1);

        // ONE case translation and ONE time loop per path, both the ones the gates call. A private
        // copy here would be the defect this file's header names.
        const RunReport r = onDevice
            ? runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true,
                                 /*fieldsOut=*/nullptr, endTime, /*tapsOut=*/nullptr, &mutableMesh, &writer)
            : runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true,
                           /*fieldsOut=*/nullptr, endTime, /*pressureTaps=*/nullptr, &mutableMesh,
                           /*alphaTaps=*/nullptr, &writer);

        if (r.steps >= nSteps && r.time < endTime - scalar(0.5)*r.deltaT)
        {
            throw std::runtime_error(
                "brae interFoam: stopped after " + std::to_string((long)r.steps) + " steps at t = "
                + std::to_string((double)r.time) + ", short of endTime " + std::to_string((double)endTime)
                + " -- the fixed-step count ran out before the clock did.");
        }
        // BRAE_PRINT_TURB_SOLVES=1: the closure's solves as OpenFOAM's log prints them, one line each -- the
        // instrument that says whether a closure field differs because a solve STOPPED somewhere else
        if (std::getenv("BRAE_PRINT_TURB_SOLVES") != nullptr)
        {
            auto printSolves = [](const char* name, const std::vector<cpu::interFoam::LinearSolveRecord>& v)
            {
                for (const cpu::interFoam::LinearSolveRecord& q : v)
                {
                    std::printf("  Solving for %s, Initial residual = %.17g, Final residual = %.17g, "
                                "No Iterations %d\n",
                                name, (double)q.initialResidual, (double)q.finalResidual, (int)q.nIterations);
                }
            };
            printSolves("alpha.water", r.alphaSolves);
            printSolves("p_rgh", r.pSolves);
            printSolves("pcorr", r.pcorrSolves);
            printSolves("epsilon", r.epsilonSolves);
            printSolves("omega", r.omegaSolves);
            printSolves("k", r.kSolves);
        }
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
