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

} // namespace interFoam
} // namespace cpu
} // namespace brae
