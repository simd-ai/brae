#pragma once
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
};

// What one step reports, on either arm.
struct StepReport
{
    scalar meanCoNum = 0;   // centralCourantNo.H, on this step's amaxSf
    scalar CoNum     = 0;
    int    status    = 0;   // see statusText()
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
    controlLimiterScalarOnV = 16
};

} // namespace rhoCentral
} // namespace brae
