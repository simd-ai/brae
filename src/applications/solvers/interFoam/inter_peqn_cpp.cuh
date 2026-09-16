#pragma once
// interFoam's pressure corrector -- the host reference.
//
// provenance:
//   openfoam:
//     file: applications/solvers/multiphase/interFoam/pEqn.H:1-89
//     also: src/finiteVolume/finiteVolume/fvc/fvcMeshPhi.C   (ddtCorr's weighting)
//   brae:
//     reference: this header
//     cuda:      (pending)
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
    scalar tolP    = 1e-7;
    scalar relTolP = 0;
    int    maxIterP = 2000;
    label  nCorrectors = 1;          // pimple.correct()
    label  nNonOrthogonalCorrectors = 0;
    // p_rgh has no value-fixing patch anywhere -> the system is singular and needs a reference.
    bool   needReference = false;
    label  pRefCell  = 0;
    scalar pRefValue = 0;
};

struct PressureStepInput
{
    const FvVectorMatrix*      UEqn      = nullptr;   // the RELAXED momentum matrix, before the force
    const std::vector<scalar>* rho       = nullptr;
    const std::vector<scalar>* gh        = nullptr;
    const std::vector<scalar>* ghf       = nullptr;   // internal faces
    const SurfaceScalarField*  stf       = nullptr;   // surfaceTensionForce, faces
    const SurfaceScalarField*  snGradRho = nullptr;
    const DdtCorrInput*        ddt       = nullptr;   // null = no ddtCorr (steady start)
};

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
