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
#include "kepsilon_coeffs.cuh"   // KEpsilonCoeffs: the case's own closure constants, carried on the field set
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
    // `RAS { turbulence off; }` -- OpenFOAM still CONSTRUCTS the model (k, epsilon, nut, alphat all
    // read, alphat MUST_READ) and rhoSimpleFoam.C:64 still calls turbulence->validate(), which runs
    // correctNut() ONCE; only the per-iteration correct() returns early (kEpsilon.C:216). So a frozen
    // case transports the validate-time nut and alphat forever. Measured on the rhoBoxF oracle: OF
    // writes nut = Cmu*k^2/eps = 0.001265625 from a 1e-3 file seed, and alphat = rho0*nut/Prt.
    bool                   turbulenceFrozen = false;
    std::string            rasModel;            // e.g. "kEpsilon"; empty when laminar
    GeometricField<scalar> k, epsilon, nut, alphat;
    // Which alphat patches carry compressible::alphatWallFunction, and each one's OWN Prt.
    //
    // OpenFOAM ends every EddyDiffusivity::correctNut() with alphat_.correctBoundaryConditions()
    // (EddyDiffusivity.C:38), and on such a patch that evaluates to operator==(rhow*tnutw/Prt_)
    // (alphatWallFunctionFvPatchScalarField.C:125). That Prt_ defaults to 0.85 (:76) and is NOT the
    // turbulence model's, which defaults to 1.0 -- one case carries two different turbulent Prandtl
    // numbers and using either everywhere is wrong somewhere.
    //
    // Gathered here because this is where 0/alphat is read, and per patch rather than per face because
    // the entry is a dictionary key. The key may be an exact name, a GROUP or a REGEX: squareBend keys
    // it as `(?i).*walls` against a patch literally named `walls`, so an exact-name compare misses it and
    // Prt silently reverts to the model's 1.0 -- the legacy solver records that as ~15% low wall alphat
    // and wall heat flux (gpuRhoSimpleFoam.cu:437-452). findPatchEntry does OpenFOAM's resolution.
    std::vector<char>   alphatWallFn;   // per patch, 1 = carries an alphat wall function
    std::vector<scalar> alphatPrt;      // per patch, that patch's own Prt (0.85 where unset)

    // THE CASE'S OWN closure coefficients and turbulent Prandtl number, read where the turbulence dict
    // is read and carried on the field set so the SOLVE uses them.
    //
    // They used to live in a local inside createFields that was used for the construction-time
    // correctNut and then discarded, while StepInput carried its own `KEpsilonCoeffs keCoeffs{}` and
    // `scalar Prt = 1.0` that NOTHING ever assigned. So a case naming `kEpsilonCoeffs { Cmu 0.1; }` got
    // 0.1 for the initial nut and 0.09 for every iteration after it, and `Prt 0.85` reached the initial
    // alphat and never the loop's. Initialisation and the loop disagreed -- which is worse than either
    // being wrong consistently, and is the shape of defect an input struct with a plausible default
    // invites. Sourcing both from the CASE removes the second copy that could drift.
    //
    // OpenFOAM reads SIX coefficients here (kEpsilon.C:199-204: Cmu, C1, C2, C3, sigmak, sigmaEps);
    // createFields read only Cmu, so the other five were the model's defaults whatever the case said.
    KEpsilonCoeffs keCoeffs{};
    scalar         Prt = 1.0;           // EddyDiffusivity's, default 1.0 (EddyDiffusivity.C:36) -- and
                                        // NOT alphatWallFunction's, whose default is 0.85. See alphatPrt.
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
