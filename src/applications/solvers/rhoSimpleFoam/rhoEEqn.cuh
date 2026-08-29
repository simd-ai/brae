#pragma once
// CUDA implementation of rhoSimpleFoam's energy equation -- the device twin of rhoEEqn_cpp.
//
// provenance:
//   openfoam:  applications/solvers/compressible/rhoSimpleFoam/EEqn.H
//   reference: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu  (gated against OpenFOAM's own
//              stage_Ekp / stage_eD / stage_eSrc dumps by tests/rho_eeqn_vs_openfoam.sh)
//   cuda:      src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu
//   tests:     tests/test_rho_eeqn_cuda.cu
//
// rhoEEqn, not EEqn: brae puts every source directory on one include path and the host references are
// already rhoUEqn_cpp / rhoPEqn_cpp / rhoEEqn_cpp for that reason. Same rule as its two siblings.
//
// THE KINETIC-ENERGY SOURCE DIFFERS BY ENERGY VARIABLE, and this is the whole of EEqn.H's branching:
//
//     he == e :  Ekp = 0.5|U|^2 + p/rho          fvc::div(phi, Ekp)
//     he == h :  K   = 0.5|U|^2                  fvc::div(phi, K)
//
// Picking one is a wrong equation for the other thermo -- not a small error, a different conservation
// law. `isE` selects it here and is required rather than defaulted, because a default would silently be
// right on half the cases.
//
// THE FACE DIFFUSIVITY IS NOT THE OWNER CELL'S. effectiveFaceViscosity interpolates alphaEff to the faces
// and then overwrites the BOUNDARY faces with the patch values. On a wall carrying an alphat wall
// function the face value differs from the owner cell's by the whole of alphat, so the boundary array is
// a required argument, exactly as in rhoUEqn.
//
// `corrected` HAS TWO HALVES AND THIS IS THE EQUATION THAT PROVED IT. correctedLaplacian switches the
// implicit coefficient to nonOrthDeltaCoeffs AND adds the explicit deferred source
// -V*div(gamma*magSf*(corrVec & interpolate(grad he))). The host reference shipped with only the implicit
// half: on angledDuct that left the source 2.14e-05 out while the DIAGONAL stayed exact at 1.63e-15, so
// every gate that compared D() passed and the defect survived until the source was compared separately.
// Both halves are here. See PORT.md.
//
// THE RELAXATION SENTINEL. relaxHe == 1.0 cannot distinguish "the case named no factor for e" (OpenFOAM
// does not relax) from "the case named 1" (OpenFOAM DOES relax, and relax(1.0) is not the identity -- it
// still applies the diagonal-dominance clamp and adds (D - D0)*psi to the source; only alpha <= 0
// early-returns). relaxEquationHe carries that distinction, as relaxEquationU does for the momentum
// equation. The host reference used the 1.0 sentinel and has been brought into line, so the two now
// agree; no validation case names a factor of exactly 1 for he, so that agreement is by construction.
//
// REFUSED, not ignored -- identical to the reference: MRF (EEqn.H adds fvc::div(MRF.phi(), p) when it is
// active) and any fvOptions type that is not ported. Refused here and NOT in the reference: a mesh with
// coupled patches, which buildDeviceMesh keeps out of the LDU entirely.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
// PressureMatrix is the assembled SCALAR matrix in this lineage -- diag/upper/lower, source, iC/bC -- and
// the energy equation produces exactly that object. Reused rather than defined a second time under
// another name: two identical structs is the hazard, not the shared name. It carries `faceFluxCorr`,
// which the energy equation never populates because nothing calls EEqn.flux(). It lives in the
// incompressible solver's header for historical reasons and belongs in a shared one.
#include "pEqn.cuh"
// cpu::rhoSimple::DivScheme -- the compressible lineage's own enum, so adding a scheme is a decision
// taken against rhoSimpleFoam's tutorials rather than inherited from simpleFoam's.
#include "rhoUEqn_cpp.cuh"
#include <string>

namespace brae {
namespace gpu {
namespace rhoSimple {

struct RhoEnergyInput
{
    // The MASS flux, kg/s. Both the implicit convection of he and the explicit kinetic-energy divergence
    // are taken against it.
    const DeviceBuffer<scalar>* phiInt = nullptr;
    const DeviceBuffer<scalar>* phiBnd = nullptr;

    // alphaEff = CpByCpv*(alpha + alphat), supplied rather than formed here: it is the thermo's and the
    // turbulence model's, not this equation's. Boundary faces take the PATCH value -- see the header.
    const DeviceBuffer<scalar>* alphaEffCell    = nullptr;
    const DeviceBuffer<scalar>* alphaEffBndFace = nullptr;

    // Fields the kinetic-energy term is built from. p and rho are needed only on the `e` branch, but both
    // are required regardless so a caller cannot half-supply the inputs of an equation it selected.
    const DeviceBuffer<scalar>* Ux   = nullptr;
    const DeviceBuffer<scalar>* Uy   = nullptr;
    const DeviceBuffer<scalar>* Uz   = nullptr;
    const DeviceBuffer<scalar>* pCell = nullptr;
    const DeviceBuffer<scalar>* rhoCell = nullptr;
    // ...and their boundary values, because fvc::div(phi, Ekp) reads Ekp on boundary faces too.
    const DeviceBuffer<scalar>* UxBnd = nullptr;
    const DeviceBuffer<scalar>* UyBnd = nullptr;
    const DeviceBuffer<scalar>* UzBnd = nullptr;
    const DeviceBuffer<scalar>* pBnd   = nullptr;
    const DeviceBuffer<scalar>* rhoBnd = nullptr;

    // he == "e" (sensibleInternalEnergy) selects Ekp; he == "h" selects K. Required, not defaulted.
    bool isE = true;

    scalar relaxHe          = 1.0;
    bool   relaxEquationHe  = false;   // see the sentinel note in the header

    // `bounded Gauss <scheme>` on each of the two convection terms. They are separate entries in
    // fvSchemes -- div(phi,e) and div(phi,Ekp) -- and a case can bound one and not the other.
    bool   boundedHe = false;
    bool   boundedKE = false;

    cpu::rhoSimple::DivScheme schemeHe = cpu::rhoSimple::DivScheme::upwind;
    cpu::rhoSimple::DivScheme schemeKE = cpu::rhoSimple::DivScheme::upwind;
    scalar gradHeLimitK = 0.0;
    scalar gradKELimitK = 0.0;

    bool   correctedLaplacian = false;   // BOTH halves -- see the header
    scalar snGradLimitCoeff   = 0.0;

    bool hasMRF            = false;   // EEqn.H adds fvc::div(MRF.phi(), p) -- refused
    bool hasFvOptions      = false;   // refused
    bool hasCoupledPatches = false;   // refused
    std::string fvOptionUnsupported;
};

// Ekp (he == e) or K (he == h), on cells and on boundary faces. Exposed on its own so it can be gated
// directly against OpenFOAM's stage_Ekp rather than only through the assembled matrix -- the same reason
// the host reference exposes it.
void kineticEnergy(
    DeviceBuffer<scalar>&    keCell,
    DeviceBuffer<scalar>&    keBnd,
    const DeviceMesh&        dm,
    const RhoEnergyInput&    in);

// EEqn.H's explicit kinetic-energy divergence, fvc::div(phi, Ekp|K), returned EXTENSIVE (V*div) because
// that is how an fvMatrix consumes an added field. Exposed so the branch's EFFECT on the equation -- not
// only the field it builds -- can be measured on its own.
void kineticEnergyDivergence(
    DeviceBuffer<scalar>&    out,
    const DeviceMesh&        dm,
    const RhoEnergyInput&    in);

// EEqn.H up to and including EEqn.relax(): the implicit convection and diffusion of he, plus the explicit
// kinetic-energy divergence in the source. Returned before solve(), which is the state OpenFOAM's
// stage_eD / stage_eSrc harness dumps -- so this and the reference can be compared against the same
// oracle without either running a linear solver.
//
// Throws (host-side) on every refusal in RhoEnergyInput, matching the reference's contract.
void assembleEEqn(
    PressureMatrix&             E,
    const DeviceMesh&           dm,
    const DeviceBoundary&       dbHe,
    const DeviceBuffer<scalar>& he,
    const RhoEnergyInput&       in);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
