#pragma once
// interFoam's pressure corrector -- the host reference.
//
// provenance:
//   openfoam:
//     file: applications/solvers/multiphase/interFoam/pEqn.H:1-89
//     also: src/finiteVolume/finiteVolume/fvc/fvcMeshPhi.C   (ddtCorr's weighting)
//   brae:
//     reference: this header
//     cuda:      src/applications/solvers/interFoam/device_inter_peqn.cu -- the four notes below;
//                the laplacian, the solve and the non-orthogonal loop are the device machinery
//                every other pressure corrector already shares.
//     tests:     tests/test_inter_peqn_cpp.cu
//
// WHAT IS HERE. The laplacian, the solve and the non-orthogonal loop are the machinery brae already has
// and shares with every other pressure corrector. THIS FILE IS THE FOUR THINGS THAT ARE interFoam's OWN,
// each of which is a place a port that copies simpleFoam's pEqn -- or interFoam's own UEqn -- is wrong.
//
// 1. phig CARRIES NO snGrad(p_rgh), WHERE UEqn's SOURCE DOES.
//
//        UEqn.H:19-27   reconstruct((stf - ghf*snGrad(rho) - snGrad(p_rgh)) * magSf)
//        pEqn.H:28-34   phig = (stf - ghf*snGrad(rho)) * rAUf * magSf
//
//    The pressure gradient is EXPLICIT in the momentum predictor and IMPLICIT here -- it is the
//    laplacian being solved. Carrying snGrad(p_rgh) into phig as well would count it twice, and the
//    result still converges: to a flow with the wrong pressure-gradient balance at the interface.
//    buoyancyFlux() below therefore takes no p_rgh at all, so the term cannot be passed by accident,
//    and the gate asserts the exact relationship between the two expressions rather than the absence.
//
// 2. ddtCorr IS WEIGHTED BY interpolate(rho*rAU), NOT BY interpolate(rho)*rAUf.
//
//        phiHbyA = fvc::flux(HbyA) + MRF.zeroFilter(fvc::interpolate(rho*rAU)*fvc::ddtCorr(U, phi, Uf))
//
//    The PRODUCT is interpolated, once. rAUf is a different field -- interpolate(rAU) -- and reusing it
//    here with a separately interpolated rho is the obvious economy. Across a VoF interface rho jumps
//    by a factor of 1000 in one face, which is exactly where interpolating a product and multiplying
//    two interpolations stop agreeing, and exactly where the answer is decided.
//
// 3. THE VELOCITY CORRECTION DIVIDES BY rAUf INSIDE reconstruct AND MULTIPLIES BY rAU OUTSIDE.
//
//        U = HbyA + rAU*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf)
//
//    Not rAU*reconstruct(phig - flux), and not reconstruct((phig - flux)) on its own. The two forms
//    coincide EXACTLY when rAU is uniform -- which it is on any fixture with a uniform mesh and a
//    single-phase momentum equation -- so a test built on such a fixture cannot tell them apart. The
//    gate below uses a deliberately non-uniform rAU for that reason, and keeps the uniform case as the
//    identity it is.
//
// 4. WHEN p_rgh NEEDS A REFERENCE, p IS SHIFTED AND p_rgh IS REBUILT FROM THE SHIFTED p.
//
//        p == p_rgh + rho*gh
//        if (p_rgh.needReference()) { p += pRefValue - p[pRefCell];  p_rgh = p - rho*gh; }
//
//    So p_rgh does not keep the value the solve gave it. Stopping after the first line leaves p at the
//    right level and p_rgh at the wrong one, and the two then disagree by a constant that shows up in
//    the next time step's momentum source. damBreak does not exercise this -- its atmosphere patch is
//    totalPressure, which fixes a value -- but 8 of the shipped tutorials have no value-fixing p_rgh
//    patch at all, and those all do.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "ldu_matrix.cuh"
#include "inter_solve_record.cuh"
#include <stdexcept>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// phig = (surfaceTensionForce() - ghf*snGrad(rho)) * rAUf * magSf, pEqn.H:28-34.
//
// Takes no p_rgh, by design -- see note 1. The caller cannot pass the term that does not belong here.
void buoyancyFlux(const std::vector<scalar>& surfaceTensionForce,
                  const std::vector<scalar>& ghf,
                  const std::vector<scalar>& snGradRho,
                  const std::vector<scalar>& rAUf,
                  const std::vector<scalar>& magSf,
                  std::vector<scalar>&       phig);

// fvc::interpolate(rho*rAU) -- the product formed per CELL and interpolated once, see note 2. Returned
// as a face field over the internal faces; the caller supplies the mesh's own linear weights.
void rhoRAUf(const std::vector<scalar>& rho,
             const std::vector<scalar>& rAU,
             const PrimitiveMesh&       m,
             const FvGeometry&          g,
             std::vector<scalar>&       out);

// U = HbyA + rAU*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf), pEqn.H:58.
//
// `faceFlux` is phig - p_rghEqn.flux() BEFORE the division; the division by rAUf happens here so that
// the two operations cannot be separated at a call site and quietly reordered.
void correctVelocity(const std::vector<vector>&              HbyA,
                     const std::vector<scalar>&              rAU,
                     const std::vector<scalar>&              faceFlux,     // internal faces
                     const std::vector<scalar>&              rAUf,         // internal faces
                     const std::vector<std::vector<scalar>>& faceFluxBnd,
                     const std::vector<std::vector<scalar>>& rAUfBnd,
                     const PrimitiveMesh&                    m,
                     const FvGeometry&                       g,
                     const std::vector<FvPatch>&             patches,
                     std::vector<vector>&                    U);

// p = p_rgh + rho*gh, pEqn.H:72. The field interFoam writes and never solves.
void staticPressure(const std::vector<scalar>& p_rgh,
                    const std::vector<scalar>& rho,
                    const std::vector<scalar>& gh,
                    std::vector<scalar>&       p);

// pEqn.H:74-83, the branch that runs when p_rgh has no value-fixing patch. BOTH fields move: p is
// shifted to put pRefValue in pRefCell, and p_rgh is then REBUILT from the shifted p -- see note 4.
void applyPressureReference(std::vector<scalar>&       p,
                            std::vector<scalar>&       p_rgh,
                            const std::vector<scalar>& rho,
                            const std::vector<scalar>& gh,
                            label                      pRefCell,
                            scalar                     pRefValue);

// ---------------------------------------------------------------------------------------------------
// fvc::ddtCorr(U, phi) -- the transient flux correction, Euler, fixed mesh.
//
//   provenance: src/finiteVolume/finiteVolume/ddtSchemes/EulerDdtScheme/EulerDdtScheme.C
//                 (fvcDdtPhiCorr: phiCorr = phi.oldTime() - (interpolate(U.oldTime()) & Sf))
//               src/finiteVolume/finiteVolume/ddtSchemes/ddtScheme/ddtScheme.C
//                 (fvcDdtPhiCoeff, the limiter on it)
//
// WHAT IT IS FOR. phi and U are separate state: the pressure corrector writes phi, and U is
// reconstructed from it, so after a step the stored flux and the flux you would get by interpolating
// the stored velocity DO NOT AGREE. That difference is real information -- it is the part of the flux
// that lives on faces and has no cell-centred representation -- and ddtCorr feeds it back into the
// next pressure equation instead of letting it be rebuilt from the smoother, interpolated field. A
// solver that drops it decouples pressure and velocity slowly and rings.
//
// THREE THINGS, all of them in the COEFFICIENT rather than the difference:
//
//   1. THE DEFAULT IS A LIMITER, NOT A CONSTANT. ddtPhiCoeff_ is -1 unless fvSchemes says otherwise
//      (ddtScheme.H:135), and that selects
//          coeff = 1 - min(|phiCorr| / (|phi| + SMALL), 1)
//      -- so the correction is switched OFF wherever it is large compared with the flux itself, which
//      is exactly where feeding it back would be unstable. A port that used a constant 1 would apply
//      it hardest where OpenFOAM applies it least.
//
//   2. IT IS ZEROED ON EVERY PATCH WHERE U FIXES A VALUE (ddtScheme.C:~275). At a wall or a
//      prescribed inlet the flux is whatever the boundary condition says, and there is no
//      inconsistency to correct; leaving the coefficient at 1 there injects a spurious flux into the
//      pressure equation at precisely the boundaries that are supposed to be prescribed.
//
//   3. THE OLD TIME, BOTH TIMES. phi.oldTime() and U.oldTime() -- not the current fields, which are
//      what the corrector is about to produce.
struct DdtCorrInput
{
    const SurfaceScalarField*  phiOld = nullptr;      // phi.oldTime()
    const std::vector<vector>* UOld   = nullptr;      // U.oldTime(), cell values
    // fvSchemes' ddtPhiCoeff. NEGATIVE (the default) selects the limiter in note 1; a non-negative
    // value is used verbatim as a constant coefficient.
    scalar ddtPhiCoeff = -1;
    scalar deltaT = 0;
};

// Returns coeff*rDeltaT*phiCorr on the internal faces, and zero on every patch where U fixes a value.
void ddtCorr(const DdtCorrInput&           in,
             const GeometricField<vector>& U,
             const PrimitiveMesh&          m,
             const FvGeometry&             g,
             const std::vector<FvPatch>&   patches,
             SurfaceScalarField&           out);

// ---------------------------------------------------------------------------------------------------
// pEqn.H end to end: rAU, HbyA, phiHbyA, the p_rgh solve, then U and phi rebuilt.
struct PressureSolveControls
{
    // solvers/p_rgh, for every corrector but the last...
    scalar tolP = 1e-7;
    scalar relTolP = 0;
    // lduMatrix::defaultMaxIter (lduMatrix.H:125)
    int maxIterP = 1000;
    // `solver PCG; preconditioner DIC;` -> brae::pcg
    bool pcgDIC = false;
    // ...and solvers/p_rghFinal for the last. See InterFields::pSolve for why there are two.
    scalar tolPFinal = 1e-7;
    scalar relTolPFinal = 0;
    int maxIterPFinal = 1000;
    bool pcgDICFinal = false;
    // pimpleControl::finalInnerIter() (pimpleControlI.H:98-111): corrPISO == nCorrPISO. The caller
    // knows which pass this is; pressureCorrector owns the non-orthogonal half of the test.
    bool finalCorrector = true;
    label  nCorrectors = 1;          // pimple.correct()
    label  nNonOrthogonalCorrectors = 0;
    // p_rgh has no value-fixing patch anywhere -> the system is singular and needs a reference.
    bool   needReference = false;
    label  pRefCell  = 0;
    scalar pRefValue = 0;
};

// Every intermediate tools/dumpInterFoam/pEqn.H writes, taken at the same point in the pass, so a gap
// in the pressure corrector can be read term by term against OpenFOAM's own numbers. Overwritten on
// every call: what is left is the LAST corrector's, which is also what the dump leaves, because it
// writes all of them into the same time directory.
struct PressureTaps
{
    std::vector<scalar> A;
    std::vector<scalar> rAU;
    std::vector<vector> HbyA;
    // internal faces only
    std::vector<scalar> rAUf;
    std::vector<scalar> phig;
    // BEFORE `phiHbyA += phig`, which is where the dump writes it
    std::vector<scalar> phiHbyA;
    std::vector<scalar> stf;
    std::vector<scalar> snGradRho;
};

// One p_rgh solve as the solver itself reports it -- see inter_solve_record.cuh.
using PressureSolveRecord = LinearSolveRecord;

// THE FLUX A PATCH'S CONDITION NAMES, by its `phi` entry: `phi` (the default) or `rhoPhi`. interFoam
// carries both, and OpenFOAM's flux-conditional conditions look theirs up BY NAME in updateCoeffs --
// three shipped tutorials write `phi rhoPhi;` on their totalPressure top. brae's conditions are told
// their flux, and were told `phi` whatever the case said. The two are not the same switch: rhoPhi is
// the ALPHA step's flux, built from the phi the time step started on, so inside a pressure corrector
// it is one corrector behind and at step one of a case at rest it is ZERO on every face. Measured on
// solitaryGrimshaw against real OpenFOAM, with the wave model's own patch values exact to 7e-17: the
// p_rgh initial residual wrong from STEP ONE and U 2.4e-05 out after thirty; 9.6e-10 with the named
// flux. Any other name is refused: brae's interFoam has no third flux to hand over.
const std::vector<scalar>& namedPatchFlux(
    const std::string& fluxName,
    std::size_t patchIndex,
    const std::string& patchName,
    const SurfaceScalarField& phi,
    const SurfaceScalarField* rhoPhi);

struct PressureStepInput
{
    const FvVectorMatrix*      UEqn      = nullptr;   // the RELAXED momentum matrix, before the force
    const std::vector<scalar>* rho       = nullptr;
    const std::vector<scalar>* gh        = nullptr;
    const std::vector<scalar>* ghf       = nullptr;   // internal faces
    // ...AND THE BOUNDARY. `phiHbyA += phig` in pEqn.H is a whole-surfaceScalarField operation, and
    // fvc::div(phiHbyA) sums the boundary faces -- so the wall's buoyancy and surface tension enter
    // the PRESSURE EQUATION'S SOURCE through it. Adding phig to the internal faces alone drops that
    // entirely, which at a contact-angle wall is the term the contact angle exists to apply. Measured
    // on capillaryRise, where momentumPredictor is off and the pressure corrector is the ONLY route
    // the surface tension has into the solution.
    const std::vector<std::vector<scalar>>* ghfBnd = nullptr;
    // the alpha step's mass flux, for a patch whose condition names `phi rhoPhi` -- see namedPatchFlux.
    // It does not change inside the pressure correctors. Null refuses such a patch.
    const SurfaceScalarField* rhoPhi = nullptr;
    const SurfaceScalarField*  stf       = nullptr;   // surfaceTensionForce, faces
    const SurfaceScalarField*  snGradRho = nullptr;
    const DdtCorrInput*        ddt       = nullptr;   // null = no ddtCorr (steady start)
    // rho's PATCH values, which totalPressure reads -- see updatePressurePatchesFromVelocity. Required
    // when p_rgh carries a totalPressure patch, and refused by name when it is missing there.
    const std::vector<std::vector<scalar>>* rhoBnd = nullptr;
    // null = no capture
    PressureTaps* taps = nullptr;
    // appended to, one record per solve; null = not kept
    std::vector<PressureSolveRecord>* solveLog = nullptr;
};

// U.correctBoundaryConditions() FOR THE FLUX-CONDITIONAL VELOCITY PATCHES, which evaluateBoundary()
// alone does not resolve. pressureInletOutletVelocity is a directionMixed: OpenFOAM's evaluate() leaves
// the patch value at patchInternalField on an outflow face and at its normal component on an inflow
// one, and brae's class keeps its STORED value through evaluate() on purpose, so this is what moves it.
// It must run wherever OpenFOAM's U.correctBoundaryConditions() does -- after the velocity correction
// in pEqn.H, AND after the momentum predictor's solve, which ends with one. The second was missing on
// the host: under `momentumPredictor yes` the first pressure corrector then read a stale U_b in
// totalPressure's 0.5*rho*|U_b|^2, and the host sat at U 9.2e-08 from OpenFOAM on a case where the
// device, which did refresh it, sat at 8.7e-12.
void updateVelocityPatchesFromCells(
    GeometricField<vector>& U,
    const std::vector<FvPatch>& patches);

// totalPressure's updateCoeffs, at the moment OpenFOAM runs it: fvm::laplacian(rAUf, p_rgh) constructs
// an fvMatrix, whose constructor calls psi.boundaryFieldRef().updateCoeffs(). For a dimPressure field
// with no psi that is (totalPressureFvPatchScalarField.C:118-127)
//
//     p_b = p0 - 0.5*rho_b*neg(phi_b)*magSqr(U_b)
//
// with rho_b, phi_b and U_b LOOKED UP as patch fields right then. brae's patch cannot look anything up,
// and interFoam never told it: the atmosphere of damBreak sat at p0 for the whole run, with no dynamic
// pressure on the faces drawing air in. Five steps from rest that is 1e-9 of p_rgh and no field gate
// could see it; what saw it was the solver log against OpenFOAM's, where the initial residual of the
// THIRD solve of step one was 1.7e-06 out with the first two exact. The flux must already have reached
// the patch (pushFluxToPatches); U's patch values and rho's are passed here.
void updatePressurePatchesFromVelocity(
    GeometricField<scalar>& p_rgh,
    const GeometricField<vector>& U,
    const std::vector<std::vector<scalar>>* rhoBnd,
    const std::vector<FvPatch>& patches);

// One pass of pEqn.H. p_rgh, U and phi are all updated in place; `p` is (re)built at the end.
void pressureCorrector(GeometricField<scalar>&      p_rgh,
                       GeometricField<vector>&      U,
                       SurfaceScalarField&          phi,
                       std::vector<scalar>&         p,
                       const PressureStepInput&     in,
                       const PressureSolveControls& sc,
                       const PrimitiveMesh&         m,
                       const FvGeometry&            g,
                       const std::vector<FvPatch>&  patches);

} // namespace interFoam
} // namespace cpu
} // namespace brae
