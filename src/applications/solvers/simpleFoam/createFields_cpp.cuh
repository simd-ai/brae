#pragma once
// _cpp REFERENCE -- host transcription of simpleFoam's createFields.H.
//
// provenance:
//   openfoam:
//     file: applications/solvers/incompressible/simpleFoam/createFields.H:1-46
//     also: src/finiteVolume/cfdTools/incompressible/createPhi.H
//           src/finiteVolume/cfdTools/general/findRefCell/findRefCell.C:33-119  (setRefCell)
//           src/OpenFOAM/fields/GeometricFields/GeometricField/GeometricField.C:1068-1085 (needReference)
//   brae:
//     reference: src/applications/solvers/simpleFoam/createFields_cpp.cu
//     tests:     tests/test_create_fields_cpp.cu
//
// OpenFOAM, in order:
//   p    MUST_READ
//   U    MUST_READ
//   phi  createPhi.H -- READ_IF_PRESENT, and when absent it is CALCULATED as fvc::flux(U)
//   setRefCell(p, simple.dict(), pRefCell, pRefValue)
//   mesh.setFluxRequired(p.name())
//   singlePhaseTransportModel / turbulenceModel::New / createMRF.H / createFvOptions.H
//
// THE phi READ IS NOT COSMETIC. `READ_IF_PRESENT` means a restart continues from the flux the previous
// run wrote, while a fresh start derives it from U. brae has already shipped a solver that always
// recomputed it, which silently changed what a restart resumed from -- the fields matched on disk and the
// trajectory did not. So the two branches are distinguished here and the test asserts BOTH.
//
// setRefCell (findRefCell.C:33-119) applies only when the field NEEDS a reference, and
// GeometricField::needReference() is "no boundary patch fixes a value" (GeometricField.C:1073-1084). It
// reads `pRefCell` / `pRefPoint` / `pRefValue` from the SIMPLE dictionary -- keys named after the FIELD,
// so `p` gives `pRefCell`. If a reference is needed and neither key is present, OpenFOAM raises a fatal
// error; this reproduces that rather than quietly picking cell 0, because "quietly picking cell 0" pins
// the pressure level somewhere the user did not choose.
//
// NOT COVERED HERE, and refused by the callers rather than skipped: the transport model, the turbulence
// model, MRF and fvOptions. They are separate manifest components.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {

struct SimpleFields
{
    GeometricField<scalar> p;
    GeometricField<vector> U;
    SurfaceScalarField     phi;
    bool                   phiWasRead = false;   // false => derived from fvc::flux(U)
    label                  pRefCell = -1;        // -1 => p does not need a reference
    scalar                 pRefValue = 0.0;
};

// p.needReference(): true when NO boundary patch fixes a value (GeometricField.C:1068-1085).
bool needReference(const GeometricField<scalar>& p);

// createFields.H + createPhi.H + setRefCell.
//   timeDir     the time directory to read from (e.g. <case>/0 or <case>/282)
//   simpleDict  the SIMPLE sub-dictionary of fvSolution, where the reference keys live
SimpleFields createFields(
    const std::string&          timeDir,
    const FoamDict*             simpleDict,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

} // namespace cpu
} // namespace brae
