#pragma once
// CUDA implementation of rhoSimpleFoam's pressure equation -- the device twin of rhoPEqn_cpp.
//
// provenance:
//   openfoam:  applications/solvers/compressible/rhoSimpleFoam/pEqn.H
//     also:    src/finiteVolume/cfdTools/general/adjustPhi/adjustPhi.C:35-140
//              src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C:1011-1023  (setReference)
//   reference: src/applications/solvers/rhoSimpleFoam/rhoPEqn_cpp.cu  (gated against OpenFOAM's own dumps,
//              BOTH branches -- tests/rho_peqn_vs_openfoam.sh runs OpenFOAM twice, transonic no and yes)
//   cuda:      src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu
//   tests:     tests/test_rho_peqn_cuda.cu -- registered four times (CMakeLists.txt): subsonic and
//              transonic, each laminar and turbulent, stage by stage against the _cpp reference.
//
// rhoPEqn, not pEqn: simpleFoam already owns pEqn.cuh and brae puts every source directory on one include
// path. Same reason as rhoUEqn.cuh.
//
// TWO BRANCHES, AND THEY ARE NOT A DETAIL. simple.transonic() selects between two different equations:
//
//   subsonic   -fvm::laplacian(rhorAUf, p) == fvc::div(phiHbyA)          after adjustPhi
//   transonic   fvm::div(phid, p) - fvm::laplacian(rhorAUf, p) == ...    after the psi*p subtraction
//
// The transonic branch also calls pEqn.relax(); the subsonic one does not. Porting one and defaulting the
// other is how a solver ends up correct on the tutorials it was debugged against and wrong everywhere else.
//
// WHAT MAKES THIS DIFFERENT FROM simpleFoam/pEqn.cu:
//   * rhorAUf is fvc::interpolate(rho*rAU) -- the RHO-WEIGHTED diffusivity, and it is interpolate OF THE
//     PRODUCT, not interpolate(rho)*interpolate(rAU). Those differ on a non-uniform rho.
//   * phiHbyA is fvc::interpolate(rho)*fvc::flux(HbyA) -- a MASS flux, so div(phiHbyA) is a mass balance.
//   * psi exists at all. The transonic branch is built entirely from it, and the closed-volume correction
//     after the solve is psi-WEIGHTED where the incompressible one is a plain volume average.
//
// ---------------------------------------------------------------------------------------------------
// WHICH LINEAR SOLVER, AND WHY IT IS NOT THE SAME ON BOTH BRANCHES.
//
// This module ASSEMBLES; the driver SOLVES. But the branch decides which solvers are legal, so the choice
// belongs in this file's contract rather than in the driver's judgement:
//
//   SUBSONIC is SYMMETRIC. Its only implicit term is fvm::laplacian, and fvm.cuh:73-77 sets
//   upper[f] == lower[f] == dc*gamma*magSf. A symmetric positive-definite operator is what CG requires, so
//   deviceAMGPCG -- and the cached AMG hierarchy below -- is valid here and is the fast path.
//
//   TRANSONIC is ASYMMETRIC. It adds fvm::div(phid, p), and fvm.cuh:388-393 sets lower[f] = -w*phi and
//   upper[f] = lower[f] + phi. Those differ by phi at every face with flow through it. CG on an asymmetric
//   matrix does not merely converge slowly -- it is solving a different problem, and it can converge
//   confidently to the wrong pressure. The transonic branch must go to BiCGStab (deviceBiCGStab /
//   pbicgstab), NOT to deviceAMGPCG.
//
// PressureMatrix::view() hands either solver the same DeviceLduView, so the switch is one branch in the
// driver on `in.transonic` -- but it is a correctness switch, not a performance one.
//
// THE AMG HIERARCHY IS A FUNCTION OF THE MESH, NOT OF THE EQUATION, so the cache is already reusable here.
// buildOrLoadAMG(fineOwner, fineNei, faceWeights, nFine, cacheDir, writeCache) takes addressing and |Sf|
// and nothing else, and device_amg.cuh:115-117 is explicit that only the STRUCTURE is cached while
// cDiag/cUpper/cLower are Galerkin-rebuilt from the current fine matrix each step. The agglomeration for
// this pressure equation is therefore bit-identical to the incompressible one on the same mesh, and a
// .brae_amgcache written by a simpleFoam run is valid for a rhoSimpleFoam run on that mesh.
//
// GAP, and it is the reason the compressible path has always been cold: NOTHING IN THE COMPRESSIBLE
// LINEAGE THREADS amgCacheDir. simpleFoam.cu:185-187 calls buildOrLoadAMG with in.amgCacheDir and
// simpleFoamV2.cu:1375 fills it from the environment; gpuRhoSimpleFoam.cu contains no reference to AMG at
// all. So every compressible run re-agglomerates from scratch. Closing that is a DRIVER change, not a
// change to this module -- this header records it so the driver is written knowing the hierarchy is
// already there to be reused. Recorded in PORT.md's open findings.
//
// ---------------------------------------------------------------------------------------------------
// REFUSED, not ignored -- identical to the reference: MRF.makeRelative, any fvOptions, and
// fixedFluxPressure (constrainPressure is not ported, and a case that names it wants a boundary condition
// this equation does not apply).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
// PressureMatrix, correctFlux, relaxField, correctVelocity and the device adjustPhi/setReference. The
// assembled object and those stages are the SAME in both lineages -- as FvScalarMatrix is on the host --
// so they are reused rather than defined a second time. Same argument as rhoUEqn.cuh makes for
// MomentumMatrix.
#include "pEqn.cuh"
// MomentumMatrix: pEqn.H consumes UEqn.A() and UEqn.H(), so the pressure equation cannot be assembled
// without the momentum matrix this solver's rhoUEqn.cu produced.
#include "rhoUEqn.cuh"
#include <string>

namespace brae {
namespace gpu {
namespace rhoSimple {

struct RhoPressureInput
{
    const DeviceBuffer<scalar>* rhoCell    = nullptr;   // nCells -- the SOLVER's relaxed rho, not thermo.rho()
    const DeviceBuffer<scalar>* rhoBndFace = nullptr;   // boundary faces, the PATCH value
    // psi = d(rho)/dp, the compressibility. The transonic branch is built from it and the closed-volume
    // correction is weighted by it, so it is required on both branches rather than transonic-only.
    const DeviceBuffer<scalar>* psiCell    = nullptr;   // nCells
    const DeviceBuffer<scalar>* psiBndFace = nullptr;   // boundary faces

    // simple.transonic(). Selects the whole branch, not a term -- and, per the header, which linear
    // solver the driver may legally use on the result.
    bool   transonic = false;

    // relaxationFactors/equations p, applied ONLY on the transonic branch because that is the only one
    // that calls pEqn.relax(). relaxPSpecified is separate because relax(1.0) is NOT the identity: it
    // still runs the diagonal-dominance clamp and adds (D - D0)*psi to the source. A case that NAMES 1
    // and a case that names nothing are different equations, and relaxP == 1.0 cannot tell them apart --
    // the same sentinel defect rhoUEqn.cuh carries relaxEquationU for.
    scalar relaxP          = 1.0;
    bool   relaxPSpecified = false;

    // pressureControl.refCell()/refValue(). -1 means the operator is not singular and needs no reference.
    label  pRefCell  = -1;
    scalar pRefValue = 0.0;

    bool   correctedLaplacian = false;   // BOTH halves, as everywhere else in this port
    scalar snGradLimitCoeff   = 0.0;

    // constrainHbyA replaces HbyA by U on a patch whose U is NOT assignable -- and assignable() is not
    // fixesValue(): slip and inletOutlet are non-assignable WITHOUT fixing a value. The distinction is
    // why fv_patch_field carries both predicates, and it cannot be recovered from the device boundary's
    // bcType, so the caller supplies the resolved per-face mask. Required.
    const DeviceBuffer<label>* takeUAtBoundary = nullptr;
    // adjustPhi's per-face `adjustable` mask: an outflow face that is neither fixed-value nor
    // inletOutlet-at-outflow. Required whenever pRefCell >= 0, which is exactly when the domain is closed
    // and the singular operator would otherwise be handed an inconsistent right-hand side.
    const DeviceBuffer<label>* adjustable = nullptr;

    bool hasMRF               = false;   // MRF.makeRelative(phiHbyA) -- refused
    bool hasFvOptions         = false;   // refused
    bool hasFixedFluxPressure = false;   // constrainPressure not ported -- refused
    bool hasCoupledPatches    = false;   // buildDeviceMesh keeps them out of the LDU -- refused
    std::string fvOptionUnsupported;
};

// Every intermediate of pEqn.H, in the order OpenFOAM produces them, kept separate so a gate can name the
// first divergent stage instead of comparing one folded number.
struct RhoPressureStages
{
    DeviceBuffer<scalar> rAU;            // 1/UEqn.A()
    DeviceBuffer<scalar> rhorAUf;        // fvc::interpolate(rho*rAU), internal faces -- the diffusivity
    DeviceBuffer<scalar> rhorAUfBnd;     // and on boundary faces, the patch value
    DeviceBuffer<scalar> HbyA[3];        // cells,  rAU*UEqn.H()
    DeviceBuffer<scalar> HbyAb[3];       // boundary faces, AFTER constrainHbyA
    DeviceBuffer<scalar> phiHbyA0Int;    // interpolate(rho)*flux(HbyA), BEFORE either branch
    DeviceBuffer<scalar> phiHbyA0Bnd;
    DeviceBuffer<scalar> phiHbyAInt;     // after the branch: psi*p subtraction, or adjustPhi
    DeviceBuffer<scalar> phiHbyABnd;
    DeviceBuffer<scalar> phidInt;        // TRANSONIC ONLY: (interpolate(psi)/interpolate(rho))*phiHbyA
    DeviceBuffer<scalar> phidBnd;

    // adjustPhi's return, SUBSONIC ONLY -- it drives the psi-weighted mass correction applied to p AFTER
    // the solve, so it is a result of this stage and not a detail of it.
    bool   closedVolume = false;
    bool   transonic    = false;
    scalar massCorr     = 1.0;
};

// Stages 1-4 of pEqn.H: rAU, rhorAUf, HbyA, phiHbyA, then the branch's own treatment of phiHbyA (phid and
// the psi*p subtraction, or adjustPhi). Solves nothing.
//
// Throws (host-side) on every refusal in RhoPressureInput, matching the reference's contract.
void pressurePredictor(
    RhoPressureStages&          st,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary&       dbP,
    const MomentumMatrix&       UEqn,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& p,
    const RhoPressureInput&     in);

// Stage 5: assemble the pressure equation for whichever branch the predictor took, relax it if the branch
// relaxes, and apply setReference.
//
// The reference is applied HERE and not by the caller because OpenFOAM applies it to the ASSEMBLED matrix
// and its effect -- source[cell] += diag[cell]*value; diag[cell] += diag[cell] -- is not recoverable
// afterwards. It is a rank fix on a singular operator, not a boundary condition.
void assemblePEqn(
    PressureMatrix&             P,
    const RhoPressureStages&    st,
    const DeviceMesh&           dm,
    const DeviceBoundary&       dbP,
    const DeviceBuffer<scalar>& p,
    const RhoPressureInput&     in);

// Stage 8 -- U = HbyA - rAU*grad(p) -- is gpu::correctVelocity (pEqn.cuh), unchanged: it operates on the
// solved p and carries no rho weighting of its own.
//
// STAGE 7 IS NOT. This note used to say gpu::correctFlux was reusable here too. It is not, on two counts.
// The sign: assemblePEqn above NEGATES the entire assembled matrix -- diag, off-diagonals, source, both
// boundary coefficient arrays and the face-flux correction -- because rhoSimpleFoam writes its pressure
// equation as `fvc::div(phiHbyA) - fvm::laplacian(...) == 0` where the incompressible solver writes
// `fvm::laplacian(...) == fvc::div(phiHbyA)`. So the flux ADDS here and SUBTRACTS there, and the host
// reference says so at rhoSimpleFoam_cpp.cu's flux block. The type: gpu::correctFlux takes the
// incompressible PressureStages, not RhoPressureStages. rhoSimpleFoam.cu carries its own
// correctFluxCompressible for both reasons.

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
