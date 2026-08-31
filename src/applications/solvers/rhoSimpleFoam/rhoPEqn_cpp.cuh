#pragma once
// _cpp REFERENCE -- host transcription of rhoSimpleFoam's pressure equation, BOTH branches.
//
// provenance:
//   openfoam:
//     file: applications/solvers/compressible/rhoSimpleFoam/pEqn.H:1-110
//     also: src/finiteVolume/cfdTools/general/adjustPhi/adjustPhi.C:35-140
//           src/finiteVolume/cfdTools/general/pressureControl/pressureControl.C (limit)
//   brae:
//     reference: src/applications/solvers/rhoSimpleFoam/rhoPEqn_cpp.cu
//     cuda:      (pending -- the whole case runs in _cpp first, see PORT.md)
//     tests:     tests/test_rho_peqn_cpp.cu
//
// TWO BRANCHES, AND THEY ARE NOT A FLAG APART. `simple.transonic()` selects between two pressure
// equations that differ in four separate ways, and a port that implements one and treats the other as a
// variation gets at least one of them wrong:
//
//                            transonic                          subsonic
//   pressure equation        fvc::div(phiHbyA)                  fvc::div(phiHbyA)
//                          + fvm::div(phid, p)                   (no convective term)
//                          - fvm::laplacian(rhorAUf, p)        - fvm::laplacian(rhorAUf, p)
//   phiHbyA                 -= interpolate(psi*p)*phiHbyA       adjustPhi(phiHbyA, U, p)
//                              /interpolate(rho)
//   pEqn.relax()            YES                                 NO
//   closedVolume            never set                           set by adjustPhi, and drives the
//                                                               psi-weighted mass correction after
//   rho.relax()             SKIPPED                             applied
//
// The `pEqn.relax()` asymmetry is the one most easily lost: it appears in the transonic branch only,
// under a comment about diagonal dominance, and relaxing a pressure equation that OpenFOAM does not relax
// (or failing to relax one it does) changes the iteration without changing the converged answer -- so it
// is invisible in exactly the comparison most people run.
//
// THE FLUX FORM, AGAIN, AND THE OTHER WAY ROUND. pEqn.H builds
//
//     phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA)
//
// -- the FACTORS interpolated separately. createFields.H, one file earlier, builds the initial flux as
// `linearInterpolate(rho*U) & Sf`, interpolating the PRODUCT. Both are OpenFOAM's, they differ on a
// non-uniform mesh, and each belongs to exactly one place. See createFields_cpp.cuh, where the same trap
// is documented from the other side.
//
// rhorAUf IS rho-WEIGHTED: `fvc::interpolate(rho*rAU)`, so the pressure laplacian carries the density.
// The incompressible solver's diffusivity is rAU alone. Using rAU here gives a pressure equation with the
// wrong units and a solution that still converges.
//
// NOT COVERED, refused rather than skipped: MRF (makeRelative), fvOptions, and constrainPressure -- which
// pEqn.H calls unconditionally and which is what makes `fixedFluxPressure` mean anything. A case using
// that BC is refused, not run with a stale gradient.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace rhoSimple {

struct PressureInput
{
    const std::vector<scalar>*              rho    = nullptr;   // cells
    const std::vector<std::vector<scalar>>* rhoBnd = nullptr;
    const std::vector<scalar>*              psi    = nullptr;   // cells
    const std::vector<std::vector<scalar>>* psiBnd = nullptr;

    // simple.transonic(). Selects the whole branch, not a term.
    bool   transonic            = false;
    // relaxationFactors/equations p. Applied ONLY on the transonic branch, because that is the only
    // branch that calls pEqn.relax().
    scalar relaxP               = 1.0;
    // Whether the case NAMES a p equation relaxation factor. fvMatrix::relax() (no argument) is guarded
    // by mesh().relaxEquation(name), so a case that does not name one is not relaxed at all -- and a
    // factor of 1 is NOT the same thing as not relaxing. relax(1.0) still runs the diagonal-dominance
    // step `D = max(|D|, sumMagOffDiag)` and adds (D - D0)*psi to the source; only the division by alpha
    // becomes trivial. Treating alpha == 1 as a no-op leaves the pressure matrix subtly wrong wherever a
    // cell was not already diagonally dominant.
    bool   relaxPSpecified      = false;
    label  pRefCell             = -1;      // pressureControl.refCell(); -1 => no reference needed
    scalar pRefValue            = 0.0;
    bool   correctedLaplacian   = false;
    scalar snGradLimitCoeff     = 0.0;
    bool   hasMRF               = false;   // MRF.makeRelative -- refused
    bool   hasFvOptions         = false;   // refused
};

// Every intermediate of pEqn.H, in the order OpenFOAM produces them.
struct PressureStages
{
    std::vector<scalar> rAU;         // 1/UEqn.A()
    SurfaceScalarField  rhorAUf;     // fvc::interpolate(rho*rAU) -- the laplacian diffusivity
    std::vector<vector> HbyA;        // constrainHbyA(rAU*UEqn.H(), U, p)
    SurfaceScalarField  phiHbyA0;    // fvc::interpolate(rho)*fvc::flux(HbyA), BEFORE either branch
    SurfaceScalarField  phiHbyA;     // after the branch: psi*p subtraction, or adjustPhi
    SurfaceScalarField  phid;        // transonic only: (interpolate(psi)/interpolate(rho))*phiHbyA
    // adjustPhi's return, SUBSONIC ONLY. Drives the psi-weighted mass correction applied to p after the
    // solve, so it is a result of this stage rather than a detail of it.
    bool                closedVolume = false;
    bool                transonic    = false;
};

// adjustPhi(phi, U, p) -- adjustPhi.C:35-140. Scales the ADJUSTABLE outflow so the boundary fluxes
// balance, and reports whether the domain is closed. Exposed because its return value is what the
// subsonic branch carries forward, and because it can throw: OpenFOAM raises a fatal error when the
// continuity error cannot be removed by adjusting the outflow, and so does this.
bool adjustPhi(
    SurfaceScalarField&           phi,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const PrimitiveMesh&          m,
    const std::vector<FvPatch>&   patches);

// Stages 1-4 of pEqn.H: rAU, rhorAUf, HbyA, phiHbyA, and then the branch's own treatment of phiHbyA
// (phid + the psi*p subtraction, or adjustPhi). Solves nothing.
PressureStages pressurePredictor(
    const FvVectorMatrix&         UEqn,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// Stage 5: assemble the pressure equation for whichever branch pressurePredictor took, relax it if the
// branch relaxes, and apply setReference.
//
// The reference is applied here rather than by the caller because OpenFOAM applies it to the ASSEMBLED
// matrix and its effect (source[cell] += diag[cell]*value; diag[cell] += diag[cell]) is not recoverable
// afterwards -- it is a rank fix on a singular operator, not a boundary condition.
FvScalarMatrix assemblePEqn(
    const PressureStages&         st,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
