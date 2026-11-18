#pragma once
#include "compressibleRAS_cpp.cuh"
#include <memory>
// rhoCentralFoam's state, shared by the host reference, the device arm and the writer.
//
// Cells are nC long. Boundary arrays are nB long, every boundary face in patch order (flat index k is mesh face
// nInternalFaces + k): the same layout as DeviceMesh's and DeviceBoundary's, since no coupled patch is accepted.
#include "cf_types.cuh"
#include "geometric_field.cuh"

#include <vector>

namespace brae {
namespace rhoCentral {

struct State
{
    std::vector<scalar> rho, rhoE, e, T, p, psi;
    std::vector<vector> rhoU, U;
    std::vector<scalar> rhoB, rhoEB, eB, TB, pB, psiB;
    std::vector<vector> rhoUB, UB;
};

// The boundary conditions of the three fields the case supplies, as brae's patch-field objects
// (fv_patch_field.cuh): the host arm evaluates through them, the device arm through DeviceBoundary built from them.
// Their internal fields are not kept current: every evaluation is handed the internal values it reads.
struct PatchFields
{
    GeometricField<vector> U;
    GeometricField<scalar> p;
    GeometricField<scalar> T;
    // compressible::turbulenceModel::New(rho, U, phi, thermo)'s case (milestone J): its fields and their patches
    std::shared_ptr<brae::compressibleRAS::Case> ras;
};

// One implicit solve's report (milestone G): OpenFOAM's `smoothSolver:  Solving for <field>, Initial residual = ..,
// Final residual = .., No Iterations n` (SolverPerformance.C:95-117)
struct SolveLine
{
    char   field[8] = {0};   // Ux, Uy, Uz, e, omega, k
    scalar initialResidual = 0, finalResidual = 0;
    int    nIterations = 0;
};

// What one step reports, on either arm.
struct StepReport
{
    scalar meanCoNum = 0;   // centralCourantNo.H, on this step's amaxSf
    scalar CoNum     = 0;
    int    status    = 0;   // see statusText()
    scalar flowMin   = 0;   // LTS (milestone I): setRDeltaT.H's "Flow time scale min/max", gMin/gMax(1/rDeltaT)
    scalar flowMax   = 0;
    int    nSolves   = 0;   // the viscous correctors' solves, in OpenFOAM's order (U's valid components, then e)
    SolveLine solves[6];   // U's valid components, e, and (RAS) omega and k
};

// Nonzero step status: what OpenFOAM would stop on, and what makes the state unphysical.
enum StatusBits : int
{
    statusNegativeSeed  = 1,   // e -> T inversion seeded with T < 0 (thermoI.H FatalError)
    statusNotConverged  = 2,   // e -> T inversion past maxIter_ (thermoI.H FatalError)
    statusNotPhysical   = 4    // rho, p or T not positive, or e not finite
};

// NEGATIVE CONTROLS, for the gates only: deliberately wrong numerics a gate must see. Selected by BRAE_CONTROL_RCF,
// announced on stderr. The first two are the device arm's; the others run on either arm.
//   upwind           every directed reconstruction with limiter 0: first-order upwind instead of vanLeer/vanLeerV
//   noPressureJump   phiEp without its aSf*p_pos - aSf*p_neg term (rhoCentralFoam.C:208)
//   untrimmed        setDeltaT.H's step without Time::adjustDeltaT: the write times are not landed on
//   laggedCourant    setDeltaT.H fed the Courant number at the step before's deltaT, not at its own
//   noTransform      U on symmetryPlane, symmetry, slip and wedge faces is the cell value (zeroGradient), unprojected
//   laggedRhoB       rhoU_b and rhoE_b on constraint patches multiplied by last step's rho_b, not the solve's rho_P
//   inletOutletInverted  inletOutlet switched on -phi: the inletValue on outflow faces, the cell on inflow ones
//   laggedPhi        inletOutlet switched on the step before's phi, not on this step's (rhoCentralFoam.C:195)
//   inletOutletNewT  T's inletOutlet evaluated on the cells' new T, as if Tw.evaluate() ran after thermo.correct()
//   noPurge          purgeWrite ignored: every written time directory kept (what stage 1 did in silence)
//   restartIndexZero     a restart drops uniform/time's `index`: timeStep writes leave OpenFOAM's global grid
//   restartGridFromZero  a restart measures the runTime/adjustable write grid from 0, not from its start time
//   restartIoExtrapolate a restart takes an inletOutlet patch's value from its cells, not from the file's `value`
//   noUniformTime        the writer leaves uniform/time out, so a restart from brae's output loses the time state
//   limiterVanLeer       every reconstruct(...) limited by vanLeer (vanLeerV on U), whatever the case names
//   limiterScalarOnV     the V limiter of rhoU built from its x component alone (NVDTVD on rhoU_x), not NVDVTVDV
//   noTauMC              viscous runs: tauMC left out -- fvc::div(tauMC) from the U corrector and Sf & tauMC from
//                        sigmaDotU (rhoCentralFoam.C:240, 250) (milestone G)
//   noSigmaDotU          viscous runs: - fvc::div(sigmaDotU) left out of the rhoE equation (rhoCentralFoam.C:258-263)
//   noDynamicHead        totalPressure and totalTemperature without the dynamic head: p_b = p0, T_b = T0 on inflow too
//   nozzleFlowInverted   totalPressure and totalTemperature switched on -phi: the dynamic head on outflow, none on inflow
//   waveNoRelax          waveTransmissive without its far-field relaxation (lInf ignored: k = 0)
//   waveVolumeFlux       waveTransmissive's wave speed from phi as if it were a volume flux (not divided by rho_b)
//   totalTemperatureOnce viscous runs: totalTemperature evaluated once a step (e.correctBoundaryConditions(), :265), not
//                        again at the e matrix's construction (fixedEnergy::updateCoeffs -> Tw.evaluate())
enum Control : int
{
    controlNone = 0,
    controlUpwind = 1,
    controlNoPressureJump = 2,
    controlUntrimmed = 3,
    controlLaggedCourant = 4,
    controlNoTransform = 5,
    controlLaggedRhoB = 6,
    controlInletOutletInverted = 7,
    controlLaggedPhi = 8,
    controlInletOutletNewT = 9,
    controlNoPurge = 10,
    controlRestartIndexZero = 11,
    controlRestartGridFromZero = 12,
    controlRestartIoExtrapolate = 13,
    controlNoUniformTime = 14,
    controlLimiterVanLeer = 15,
    controlLimiterScalarOnV = 16,
    controlNoTauMC = 17,
    controlNoSigmaDotU = 18,
    controlNoDynamicHead = 19,
    controlNozzleFlowInverted = 20,
    controlWaveNoRelax = 21,
    controlWaveVolumeFlux = 22,
    controlTotalTemperatureOnce = 23,
    // milestone I: the viscous solves' loop residual rounded product by product (x86-64 OpenFOAM's form), where the
    // reference, OpenFOAM's aarch64 build, fuses each multiply-subtract (rhoCentralFoam.cu)
    controlResidualUnfused = 24,
    // milestone I, each deliberately wrong:
    controlLtsUnsmoothed = 25,         // setRDeltaT.H without fvc::smooth
    controlJanafLowOnly = 26,          // janaf's low coefficients at every T (no switch at Tcommon)
    controlAlphaheOldT = 27,           // alphahe()'s CpByCpv on the T_b of the thermo.correct(), not the matrix's
    controlMaxwellNoCreep = 28,        // maxwellSlipU without its thermal creep term
    controlMaxwellNoCurvature = 29,    // maxwellSlipU without its curvature (n & tauMC) term
    controlSmoluchowskiNoJump = 30,    // smoluchowskiJumpT with valueFraction 1: T_b = Twall
    // milestone J (RAS), each deliberately wrong:
    controlRasLaminarMuEff = 31,       // :220's muEff without mut: the laminar mu only
    controlRasNoAlphat = 32,           // :278's alphaEff without alphat: CpByCpv*alpha only
    controlRasCorrectFirst = 33,       // turbulence->correct() before the U corrector (:234), on the inviscid U, not :290
    controlRasPhiUndivided = 34,       // the model's phi() the mass flux itself, not phi/fvc::interpolate(rho)
    // milestone K (mesh motion), each deliberately wrong:
    controlMotionFluxAbsolute = 35,    // the fluxes not made relative to the mesh motion (no meshPhi at :126-133, :213-218)
    controlMotionDdtStatic = 36,       // fvm::ddt's old-time source on V, not V0
    controlMotionWallStatic = 37,      // movingWallVelocity's Uwall left at zero
    controlMotionPiovStatic = 38       // pressureInletOutletVelocity never updated (its construction typing kept)
};

} // namespace rhoCentral
} // namespace brae
