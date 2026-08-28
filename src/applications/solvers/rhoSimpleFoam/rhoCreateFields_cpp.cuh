#pragma once
// _cpp REFERENCE -- host transcription of rhoSimpleFoam's createFields.H and createFieldRefs.H.
//
// provenance:
//   openfoam:
//     file: applications/solvers/compressible/rhoSimpleFoam/createFields.H:1-60
//     also: applications/solvers/compressible/rhoSimpleFoam/createFieldRefs.H:1   (psi)
//           src/finiteVolume/cfdTools/compressible/compressibleCreatePhi.H:31-42  (phi)
//           src/finiteVolume/cfdTools/general/pressureControl/pressureControl.C:33-190
//           src/finiteVolume/cfdTools/general/findRefCell/findRefCell.C:33-119    (setRefCell)
//           src/thermophysicalModels/basic/fluidThermo/fluidThermo.C              (fluidThermo::New)
//   brae:
//     reference: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu
//     tests:     tests/test_rho_create_fields_cpp.cu
//
// TRANSCRIBED, NOT COPIED -- see PORT.md. simpleFoam's createFields_cpp is the structural model for how a
// brae `_cpp` component is shaped; none of its CONTENT is reused, because the compressible field set is a
// different set built in a different order.
//
// OpenFOAM, in order (createFields.H):
//   fluidThermo::New(mesh)                       runtime-selected; reads constant/thermophysicalProperties
//   thermo.validate(executable, "h", "e")        REFUSES any other energy variable
//   p    = thermo.p()                            a REFERENCE into the thermo, not an independent field
//   rho  READ_IF_PRESENT, else thermo.rho()
//   U    MUST_READ
//   phi  compressibleCreatePhi.H                 READ_IF_PRESENT, else linearInterpolate(rho*U) & Sf
//   pressureControl(p, rho, simple.dict())
//   mesh.setFluxRequired(p.name())
//   turbulence, initialMass = fvc::domainIntegrate(rho), createMRF.H, createFvOptions.H
// and createFieldRefs.H is one line: psi is a const reference to thermo.psi().
//
// THREE THINGS HERE ARE EASY TO GET WRONG, and all three have a history in this repo.
//
// 1. THE FLUX FORM. compressibleCreatePhi.H is `linearInterpolate(rho*U) & Sf` -- it interpolates the
//    PRODUCT. That is NOT the same as `interpolate(rho)*flux(U)`, which is what brae's fvc::rhoFlux
//    computes and what rhoSimpleFoam's own pEqn.H uses one file later for phiHbyA
//    (`fvc::interpolate(rho)*fvc::flux(HbyA)`). Linear interpolation of a product is not the product of
//    the interpolations on a non-uniform mesh, so the two differ from the first face. OpenFOAM really does
//    use both forms, in the same solver, for different quantities. This builds rho*U per cell and fluxes
//    THAT, so the initial mass flux is the one OpenFOAM starts from.
//
// 2. rho IS READ_IF_PRESENT. A restart continues from the density the previous run wrote; a cold start
//    computes it from the equation of state. brae has shipped this backwards before -- the fields matched
//    on disk and the trajectory did not -- so both branches are distinguished here and `rhoWasRead`
//    records which one ran, for the test to assert.
//
// 3. p IS THE THERMO'S FIELD. In OpenFOAM `p` is a reference into the thermo, so p and T are read by the
//    thermo constructor and are both MUST_READ. Treating p as an independent field that the thermo merely
//    consults is what lets p and psi drift apart by an iteration -- the defect the rhotiming gate exists
//    for. Here they are read together and psi is derived from the SAME T, in one place.
//
// NOT COVERED HERE, and refused by the callers rather than skipped: the turbulence model, MRF and
// fvOptions. They are separate manifest components (manifest/rhoSimpleFoam.yaml).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_dict.cuh"
#include "thermo_types.cuh"
#include "fvc.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
// rhoSimpleFoam's own components. Namespaced because simpleFoam has an assembleUEqn, a createFields and
// a pEqn of its own: the two solvers transcribe DIFFERENT OpenFOAM files that happen to share names, and
// letting them collide in one namespace would make which one a call site gets an accident of includes.
namespace rhoSimple {

// pressureControl.C:33-190. Constructed from p, rho and the SIMPLE dictionary; `limit()` is applied to p
// after the pressure solve and reports whether it clipped, because a clip requires the caller to
// re-evaluate p's boundary conditions.
//
// The limits are resolved in OpenFOAM's own order and that order is not cosmetic: pMax/pMin given
// TOGETHER short-circuit the whole boundary scan, whereas pMaxFactor / rhoMax are relative to a reference
// pressure taken from the value-fixing patches, and OpenFOAM raises a fatal error when they are asked for
// and no patch fixes a value. Defaulting instead of refusing there would scale the limit off a pressure
// nobody set.
struct PressureControl
{
    label  refCell   = -1;      // -1 => p does not need a reference
    scalar refValue  = 0.0;
    scalar pMax      = 0.0;
    scalar pMin      = 0.0;
    bool   limitMaxP = false;
    bool   limitMinP = false;

    // pressureControl::limit -- clamps p in place, returns true when either limit is active (OpenFOAM
    // returns true on `limitMaxP_ || limitMinP_`, NOT on whether a value actually moved).
    bool limit(std::vector<scalar>& p) const;
};

struct RhoSimpleFields
{
    ThermoCoeffs           thermo;
    std::string            heName;             // "h" or "e" -- thermo.validate(.., "h", "e")

    GeometricField<scalar> p;                  // thermo.p()
    GeometricField<scalar> T;
    GeometricField<vector> U;
    GeometricField<scalar> rho;
    SurfaceScalarField     phi;

    std::vector<scalar>    psi;                // createFieldRefs.H -- thermo.psi(), d(rho)/d(p) at fixed T
    // psi's BOUNDARY values. pEqn.H's transonic branch interpolates psi to faces, so the patch values are
    // part of the field, not an afterthought -- taking the owner cell's value there would silently change
    // phid on every boundary face.
    std::vector<std::vector<scalar>> psiBnd;
    // thermo.he(), the variable EEqn transports -- a FIELD, not an array, because it has boundary
    // conditions of its own. OpenFOAM derives them from T's via basicThermo::heBoundaryTypes():
    // fixedValue -> fixedEnergy, zeroGradient -> gradientEnergy, inletOutlet -> mixedEnergy. brae maps
    // only the cases where that mapping is EXACT for this thermo and refuses the rest by name; see
    // createFields_cpp.cu.
    GeometricField<scalar> he;

    bool                   rhoWasRead = false; // false => computed from the equation of state
    bool                   phiWasRead = false; // false => interpolate(rho*U) & Sf
    scalar                 initialMass = 0.0;  // fvc::domainIntegrate(rho) = sum(rho*V)

    PressureControl        pressureControl;

    // The RAS fields, when the case selects a model. OpenFOAM's compressible::turbulenceModel::New reads
    // them as part of constructing the model, so they belong to createFields even though nothing in
    // createFields.H names them: `turbulence` is built there and it is what owns k, epsilon and nut.
    // alphat is the EddyDiffusivity's, and it is read rather than derived because a restart resumes from
    // the value OpenFOAM wrote.
    bool                   turbulent = false;   // momentumTransport simulationType RAS
    std::string            rasModel;            // e.g. "kEpsilon"; empty when laminar
    GeometricField<scalar> k, epsilon, nut, alphat;
    // kOmegaSST transports omega where kEpsilon transports epsilon. Which one is populated follows
    // rasModel, and the driver branches on the same name rather than on which field happens to exist.
    GeometricField<scalar> omega;

    // heRhoThermo's STORED rho_, and the reason it exists separately from `rho` above.
    // psiThermo::rho() returns p_*psi_, recomputed from whatever p is when it is called.
    // rhoThermo::rho() returns rho_ (rhoThermo.C:233), a field heRhoThermo::calculate() fills with
    // mixture_.rho(pCells, TCells) (heRhoThermo.C:88) -- and calculate() runs in thermo.correct(),
    // which rhoSimpleFoam calls at the END OF EEqn.H, before the pressure equation. So pEqn.H's
    // `rho = thermo.rho()` hands back a density built from the PRE-SOLVE pressure, and on a case where
    // p moves hard in one iteration that is nothing like the post-solve value.
    std::vector<scalar>              rhoThermo;     // filled by thermoCorrect(), read by updateRho()
    std::vector<std::vector<scalar>> rhoThermoBnd;
};

// createFields.H + compressibleCreatePhi.H + createFieldRefs.H + pressureControl.
//   timeDir     the time directory to read from (e.g. <case>/0 or <case>/282)
//   caseDir     the case root, for constant/thermophysicalProperties
//   simpleDict  the SIMPLE sub-dictionary of fvSolution, where the reference and limit keys live
RhoSimpleFields createFields(
    const std::string&          timeDir,
    const std::string&          caseDir,
    const FoamDict*             simpleDict,
    const FoamDict*             fvSolution,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
