#pragma once
// rhoCentralFoam's case: the controls OpenFOAM reads (controlDict, fvSchemes' fluxScheme and reconstruct entries,
// fvSolution, thermophysicalProperties, turbulenceProperties), the mesh, and every start-up refusal.
//
// What runs (everything else is refused here, by name, before a field is read): single GPU, static mesh,
// hePsiThermo + pureMixture + const + hConst + perfectGas + sensibleInternalEnergy, laminar with every mu = 0
// (inviscid), Euler with a fixed deltaT or adjustTimeStep (setDeltaT.H on the central Courant number), any purgeWrite
// (milestone D), fluxScheme Kurganov or Tadmor (readFluxScheme.H), reconstruct(rho|T) vanLeer, vanAlbada, Minmod, MUSCL,
// SuperBee, limitedLinear k or upwind and reconstruct(U) their V forms or upwind (milestone F), Gauss linear gradients; on U, p and T the patch fields zeroGradient, fixedValue, slip, inletOutlet on the
// flux `phi` (milestone D), and the constraint types empty, symmetryPlane, symmetry and wedge (milestone C) on their
// own mesh patch types.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "solution_directions.cuh"
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include <memory>
#include "rhoCentralThermo.cuh"
#include "rhoCentralPatchUpdate.cuh"
#include "run_clock.cuh"
#include "time_controls.cuh"
#include "foam_dict.cuh"

#include <optional>

#include <string>
#include <vector>

namespace brae {
namespace rhoCentral {

// interpolationSchemes `linear` for a reconstruct(...) entry -- and `default linear` with no reconstruct entries, which
// is what 19 of the 19 density-based fvSchemes of the other v2412 solvers write. directionInterpolate.H then builds the
// plain linear scheme, whose weights are the mesh's (linear.H) whatever the direction: both the `pos` and the `neg`
// state are the linear interpolate. A limiter of exactly 1 in limitedSurfaceInterpolationScheme::weights gives those
// weights to the bit (1*w + 0*pos0 = w), so the flux paths take it as one more limiter selector, rhoCentral's own,
// beside device_mesh.cuh's (kVanLeerTwoByk and its kin, all in [-5, 0)).
constexpr scalar kLinearReconTwoByk = -50.0;

// fvSolution's controls of one implicit solve (milestone G): smoothSolver with the GaussSeidel or symGaussSeidel
// smoother, read with lduMatrix::solver's and smoothSolver's defaults (lduMatrixSolver.C readControls: maxIter 1000,
// minIter 0, tolerance 1e-6, relTol 0; smoothSolver.C: nSweeps 1)
struct SmoothSolveControls
{
    scalar tolerance = 1e-6, relTol = 0;
    int    maxIter = 1000, minIter = 0, nSweeps = 1;
    bool   symmetric = false;    // symGaussSeidel
    std::string smoother = "GaussSeidel";
};

struct Case
{
    std::string caseDir;
    std::string startName;    // the start time directory, as Time resolves it
    scalar      startTime = 0;
    int         writePrecision = 6;
    int         purgeWrite = 0;         // controlDict purgeWrite (TimeIO.C:322-333; negative read as 0, warned)
    TimeControls timeControls;          // adjustTimeStep, maxCo, maxDeltaT (createTimeControls.H)
    std::optional<RunClock> clock;      // the clock at the start: deltaT, write cadence, endTime (Time::setControls)

    PrimitiveMesh        m;
    FvGeometry           g;
    std::vector<FvPatch> patches;
    SolutionDirections   sd;
    GasCoeffs            gas;
    bool                 tadmor = false;   // fluxScheme Tadmor (readFluxScheme.H); Kurganov otherwise
    // local time stepping (milestone I): createRDeltaT.H's LTS = fv::localEulerDdt::enabled(mesh), ddtSchemes default
    // localEuler. setRDeltaT.H: rDeltaT = max(1/maxDeltaT, surfaceSum(amaxSf)/((2*maxCo)*V)), then fvc::smooth with
    // controlDict's rDeltaTSmoothingCoeff (default 0.02); Time advances by the controlDict deltaT, never set
    bool                 lts = false;
    // compressible RAS (milestone J): turbulenceProperties simulationType RAS; the model is read with the fields
    // (createFields: compressibleRAS::readCase)
    bool                 ras = false;
    scalar               rDeltaTSmoothingCoeff = 0.02;
    // interpolationSchemes reconstruct(rho), reconstruct(T) (rPsi, e, c) and reconstruct(U) (rhoU): the limiter of each,
    // as device_limiter.cuh's limiterOfR selects it (device_mesh.cuh's sentinels; limitedLinear's 2/max(k,SMALL) > 0),
    // and the statement as the case wrote it (milestone F)
    scalar               limRho = -1.0, limT = -1.0, limU = -1.0;   // kVanLeerTwoByk
    std::string          reconRho = "vanLeer", reconT = "vanLeer", reconU = "vanLeerV";

    // flat boundary addressing: every boundary face in patch order (no coupled patch is accepted, so flat index k is
    // mesh face nInternalFaces + k), empty faces included and flagged
    label nC = 0, nIF = 0, nB = 0;
    std::vector<label> bCell;      // owner cell of boundary face k
    std::vector<int>   bPatch;     // patch of boundary face k
    std::vector<char>  bEmpty;     // 1 on an empty patch's face

    // the transform patches (milestone C), face by face; see rhoCentralPatchUpdate.cuh for what each one changes
    std::vector<char>   bRhoConstraint;   // 1: the mesh patch is symmetryPlane, symmetry or wedge
    std::vector<char>   bTseedOwn;        // 1: T's patch field is a transform (its energy BC is the same transform)
    std::vector<int>    bUTransform;      // transformNone, transformMirror (symmetry kinds, slip) or transformRotate
    std::vector<tensor> bUR;              // I - 2.0*sqr(nHat) for a mirror, the wedge's faceT for a rotation

    // inletOutlet (milestone D): which fields carry one anywhere (each such field's patches are switched on this
    // step's phi before they are evaluated), and the faces where any of U, p, T does (the inflow diagnostic)
    bool ioU = false, ioP = false, ioT = false;
    std::vector<char>   bInletOutlet;

    // the viscous correctors (milestone G): createFieldRefs.H's inviscid = !(max(mu) > 0). When viscous, the U and e
    // solves' controls; gas.mu and gas.rPr carry the transport, gasAlphaEff(gas) the e equation's diffusivity
    bool                viscous = false;
    SmoothSolveControls solveU, solveE;

    // the nozzle boundaries (milestone H), as flat boundary-face lists with their per-face references: totalPressure
    // (the psi form) and waveTransmissive on p, totalTemperature on T. The CUDA arm runs them; the host reference
    // refuses them.
    std::vector<label>  tpFaces, wtFaces, ttFaces;
    std::vector<scalar> tpP0, tpGamma, wtFieldInf, wtLInf, wtGamma, ttT0, ttGamma;
    bool nozzleBoundaries() const { return !tpFaces.empty() || !wtFaces.empty() || !ttFaces.empty(); }

    // the rarefied-gas walls of rhoCentralFoam/BCs (milestone I, biconic25-55Run35), flat boundary-face lists:
    // maxwellSlipU on U (a partialSlip whose refValue and valueFraction it sets from mu, rho, psi, T, grad(T) and
    // tauMC), smoluchowskiJumpT on T (a mixed one: valueFraction from mu, rho, psi, Pr; refValue Twall). CUDA arm.
    std::vector<label>  msFaces, sjFaces;
    std::vector<scalar> msAcc, msVf0, sjAcc, sjTwall, sjGamma;
    std::vector<vector> msUwall, msRef0;
    std::vector<int>    msCreep, msCurv;
    scalar              sjPr = 0;
    bool slipWalls() const { return !msFaces.empty() || !sjFaces.empty(); }

    // mesh motion (milestone K): constant/dynamicMeshDict's dynamicMotionSolverFvMesh (brae's shared
    // DynamicMotionSolverFvMesh, attached to m, g and patches by the driver once the Case is in place; null: a static
    // mesh); U's movingWallVelocity faces (flat boundary indices: their fixedValue set from the motion), whether U has
    // pressureInletOutletVelocity faces (the shared directionMixed update), and which totalPressure faces take the
    // rho form (psi none: p0 - 0.5*rho_b*neg(phi)*|U_b|^2) rather than milestone H's psi form
    std::shared_ptr<DynamicMotionSolverFvMesh> motion;
    std::vector<label> mwFaces;
    bool piovU = false;
    std::vector<char> tpRho;

    // the dictionaries every reader of the case asks its questions of, kept for the unread-entry audit (dict_audit.cuh):
    // one object per file, so an entry one reader takes is not reported as unread by another's copy
    std::shared_ptr<FoamDict> fvSolution, turbulenceProperties, thermophysicalProperties;
};

// Reads and checks the case; refuses everything outside this stage's scope by name, before any field is read.
Case readRhoCentralCase(const std::string& caseDir);

// THE UNREAD-ENTRY AUDIT (dict_audit.cuh's auditDict, `brae NOTICE [unread]` lines): every entry of fvSolution,
// turbulenceProperties and thermophysicalProperties that no reader asked the case's own object for, and fvSchemes'
// top-level blocks and entries that no reader of the solver takes. Called once the fields and the RAS model are
// built, so every reader has asked its questions. It reports and does not refuse. BRAE_DICT_AUDIT=0 silences it.
void auditRhoCentralDictionaries(const Case& cs);

} // namespace rhoCentral
} // namespace brae
