#pragma once
// brae's interFoam as a RUNNABLE SOLVER -- the time loop with every component wired to it.
//
// provenance:
//   openfoam: applications/solvers/multiphase/interFoam/interFoam.C:90-176
//   brae:     the stage ORDER is inter_solve_cpp.cuh's runTimeStep, which this fills in; the case-to-
//             fields translation is inter_case_cpp.cuh's buildInterFields, which the gate also calls.
//
// WHY THE DRIVER OWNS NO NUMERICS. Every stage below is one call into a component that has its own
// gate: alphaEqnStep, momentumPredictor, pressureCorrector, alphaCourantNo, setDeltaTVoF. What this
// file adds is the STATE -- which fields are carried between steps and which are rebuilt -- and that
// is the part interFoam.C itself is mostly about.
//
// THE OLD-TIME SET IS THE CONTENT OF THIS FILE. Four fields are carried from one step to the next and
// each is read by something different:
//
//   alpha1.oldTime()  the sub-cycle's starting value, and RESTORED after it so the PIMPLE outer loop's
//                     second pass starts from the same place (inter_solve_cpp.cuh, note d)
//   U.oldTime()       fvm::ddt(rho, U)'s source, and ddtCorr's interpolated flux
//   rho.oldTime()     fvm::ddt's SOURCE while the diagonal takes the new rho -- they differ by the
//                     density ratio in every cell the interface crossed (inter_ueqn_cpp.cuh, note 1)
//   phi.oldTime()     ddtCorr's stored flux, the half that has no cell-centred representation
//
// Losing any one of them leaves a solver that runs and is wrong in a way no single equation's gate
// would catch, because each is correct in isolation.
#include "cf_types.cuh"
#include "inter_case_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include <string>

namespace brae {
namespace cpu {
namespace interFoam {

struct RunReport
{
    label  steps        = 0;
    scalar time         = 0;
    scalar deltaT       = 0;
    scalar CoNum        = 0;
    scalar alphaCoNum   = 0;
    scalar alphaMin     = 0;
    scalar alphaMax     = 0;
    scalar maxU         = 0;
    scalar worstDivPhi  = 0;
    scalar alphaMass    = 0;      // sum(alpha*V), which a closed domain must conserve
    // every p_rgh solve of the run, in order: nCorrectors x (nNonOrthogonalCorrectors + 1) per step
    std::vector<PressureSolveRecord> pSolves;
    // ...and every alpha pre-solve: one per sub-cycle per step, and none on a case without MULESCorr
    std::vector<LinearSolveRecord> alphaSolves;
    // ...and the momentum predictor's, per component -- Ux, Uy, Uz -- when the case runs one. A
    // component the mesh does not solve (the empty direction of a 2-D case) stays empty.
    std::vector<LinearSolveRecord> uSolves[3];
    // ...and the closure's two, one each per time step, on a RAS case: epsilon first, as kEpsilon.C
    // solves them
    std::vector<LinearSolveRecord> epsilonSolves;
    // kOmegaSST's first solve in place of epsilon's
    std::vector<LinearSolveRecord> omegaSolves;
    std::vector<LinearSolveRecord> kSolves;
    // ...and, where p_rgh names GAMG, what OpenFOAM prints under its own debug switches: the
    // hierarchy (cells, faces and lduAddressing::band()'s profile per level, the mesh first) and the
    // coarsest-level solve of every V-cycle of every GAMG solve, in order
    struct GamgLevel
    {
        label nCells = 0;
        label nFaces = 0;
        scalar profile = 0;
    };
    std::vector<GamgLevel> gamgLevels;
    std::vector<LinearSolveRecord> gamgCoarsestSolves;
    // ...and every pcorr solve: initCorrectPhi's at the start, then CorrectPhi's after each mesh
    // update under `correctPhi`, one per non-orthogonal pass
    std::vector<LinearSolveRecord> pcorrSolves;
    // WHICH closure produced them. The device loop can run either (BRAE_INTER_HOST_CLOSURE), and a
    // gate comparing the two has to know each arm took the path it is named for.
    bool turbulenceOnDevice = false;
};

// THE MESH A MOVING CASE IS ALLOWED TO MOVE. runInterFoam takes its mesh by const reference, as every
// gate hands it in, and a case whose mesh moves needs the SAME objects mutable -- the points, the
// geometry built from them, and the patches' copies of it, which every patch field references. The
// caller passes them here as well; the driver checks they are the objects it was given and refuses a
// moving case without them, rather than moving a copy the fields cannot see.
struct MutableMesh
{
    PrimitiveMesh* m = nullptr;
    FvGeometry* g = nullptr;
    std::vector<FvPatch>* patches = nullptr;
};

// Run `nSteps` of interFoam on a prepared case. Returns the state at the end; `verbose` prints the
// per-step line OpenFOAM's own solver prints.
RunReport runInterFoam(
    const std::string& caseDir,
    const std::string& startDir,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    label nSteps,
    bool verbose = true,
    // The FIELDS at the end, for a gate that has to compare them against
    // OpenFOAM's own. A solver's real output is files; this exists so the
    // comparison does not have to write and re-read them.
    InterFields* fieldsOut = nullptr,
    // OpenFOAM's OWN loop bound, Time::run(): `value() < endTime - 0.5*deltaT`,
    // tested BEFORE setDeltaT with the step the last iteration left in force. With
    // adjustTimeStep a step count alone cannot express it -- deltaT is not known
    // ahead of time -- and brae_interFoam ran damBreak's `endTime 0.004` out to
    // t = 0.054 before this was a parameter. Left at VGREAT, nSteps is the bound,
    // which is what every gate wants.
    scalar endTime = scalar(1.0e300),
    // The pressure corrector's intermediates, for comparing against tools/dumpInterFoam. Null in
    // every production run.
    PressureTaps* pressureTaps = nullptr,
    // see MutableMesh: required by a case whose mesh moves, ignored by one whose mesh does not
    const MutableMesh* mutableMesh = nullptr);

// ...and the SAME run on the GPU. Every operator, every corrector and every loop is the device code
// gated in tests/test_device_inter_dambreak_alpha.cu, which tracks this host driver on damBreak's own
// mesh to alpha 7.3e-11, U 2.6e-09 relative and p_rgh 6.9e-11 over five steps.
//
// THE HOOKS LIVE HERE AND NOT IN THE GATE, and that is the point. A device path whose boundary
// evaluation is written inside a test is the defect braeInterFoam.cu's own header names: the gate
// proves the step and the driver feeds it something else, with nothing comparing the two. What the
// gate exercises is this function.
//
// WHAT IS STILL ON THE HOST is what has been throughout: the boundary conditions. alpha's patch values
// and its contact angle, fvm::div's per-patch coefficients for the MULESCorr pre-solve, U's patch
// values for the stress and for constrainHbyA, and p_rgh's for the pressure laplacian. Everything that
// scales with the CELL COUNT runs on the device.
RunReport runInterFoamDevice(
    const std::string& caseDir,
    const std::string& startDir,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    label nSteps,
    bool verbose = true,
    InterFields* fieldsOut = nullptr,
    scalar endTime = scalar(1.0e300));

} // namespace interFoam
} // namespace cpu
} // namespace brae
