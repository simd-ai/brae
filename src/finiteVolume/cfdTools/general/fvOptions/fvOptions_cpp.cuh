#pragma once
// _cpp REFERENCE -- OpenFOAM's fvOptions framework, plus explicitPorositySource/DarcyForchheimer.
//
// provenance:
//   openfoam:
//     fvOption.H / fvOptionList.C        the option list and its three hooks
//     cellSetOption.C                    selectionMode / cellZone / cellSet / all
//     explicitPorositySource.C:addSup    porosityEqn built, then `eqn -= porosityEqn`
//     porosityModel.C:addResistance      transformModelData then correct(UEqn)
//     DarcyForchheimer.C:correct         kinematic UEqn -> apply(..., one, nu, U)
//     DarcyForchheimerTemplates.C:apply  the resistance itself
//     DarcyForchheimer.C:calcTransformModelData   D = csys(diag(d)),  F = csys(diag(0.5*f))
//   brae:
//     reference: src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cu
//     tests:     tests/test_fvoptions_cpp.cu, tests/fvoptions_vs_openfoam.sh
//
// THIS IS A FRAMEWORK PLUS ONE SOURCE, and the split is deliberate. ofscan counts 46 fv::option
// implementations; the framework (dictionary, cell selection, the three hooks UEqn.H and pEqn.H call)
// is shared by all of them, and each source is separate work. Reading the framework as "fvOptions is
// supported" would be exactly the silent-substitution failure this port exists to prevent, so an option
// whose `type` is not implemented is REFUSED BY NAME rather than skipped.
//
// THE THREE HOOKS, from simpleFoam's own text:
//     UEqn.H:11   == fvOptions(U)                 a source/sink on the momentum equation
//     UEqn.H:17   fvOptions.constrain(UEqn)       setValues-style constraints
//     UEqn.H:23   fvOptions.correct(U)            post-solve field manipulation
// Only the first is implemented here; the other two are refused when an option needs them.
//
// THE SIGN, which passes through two negations and is worth stating once. explicitPorositySource builds
// a porosityEqn and does `eqn -= porosityEqn` into the fvOptions matrix; simpleFoam then writes
// `UEqn == fvOptions(U)`, i.e. UEqn - optionsEqn. The two negations cancel, so the NET effect on the
// momentum matrix is the porosity equation as written:
//     diag[c]   += V[c]*isoCd
//     source[c] -= V[c]*((Cd - I*isoCd) & U[c])
// A port that applied only one of the negations gets a porosity that ACCELERATES the flow.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace fvOptions {

// One fv::option. Only explicitPorositySource/DarcyForchheimer is implemented; `unsupported` carries the
// type name of anything else so the caller can refuse it by name.
struct Option
{
    // A rotorDiskSource: implemented, but its parameters and geometry come from readFvOptions
    // and the mesh rather than from this list. Recorded so the driver knows one is present.
    bool rotorDisk = false;
        bool actuationDisk = false;   // actuationDiskSource: built from readFvOptions, not from here

    std::string        name;
    std::string        type;
    bool               active = true;
    std::vector<label> cells;                 // resolved by cellSetOption's rules; empty => all cells
    bool               allCells = false;
    std::string        unsupported;           // non-empty => this option's type is not implemented

    // DarcyForchheimer, already transformed into the global frame with the 0.5 folded into F.
    tensor D{};
    tensor F{};

    // fixedCoeff (porosityModels::fixedCoeff), the OTHER explicitPorositySource model. Its resistance is
    //     Cd = rho*(alpha + beta*|U|),  Udiag += V*tr(Cd),  Usource -= V*((Cd - I*tr(Cd)) & U)
    // (fixedCoeff.C:apply). alpha and beta are diagonal in the coordinate system's frame and transformed
    // into the global one exactly as d and f are. `rho` is NOT the local density: fixedCoeff::correct
    // reads `rhoRef` from the dict when the equation is force-dimensioned and uses 1 otherwise, which is
    // the same dispatch-on-dimensions DarcyForchheimer makes.
    // fvOptions CONSTRAINTS. These are not source terms: OpenFOAM applies them as eqn.setValues(cells,
    // values), which forces the value in those cells AND removes the coupling from their neighbours'
    // equations. Overwriting the field after the solve is NOT the same thing -- the neighbours would have
    // been solved against the unconstrained value.
    //   fixedTemperatureConstraint (mode uniform): setValues(cells, he(p, Tuniform, cells))
    //   FixedValueConstraint<scalar>:              setValues(cells, fieldValues[field])
    enum class Constraint { none, fixedTemperature, scalarFixedValue };
    Constraint constraint = Constraint::none;
    scalar     Tuniform   = 0.0;
    std::vector<std::pair<std::string, scalar>> fieldValues;

    bool   fixedCoeff = false;
    tensor alpha{};
    tensor beta{};
    scalar rhoRef = 1.0;
};

struct OptionList
{
    std::vector<Option> options;
    bool empty() const { return options.empty(); }
    // The first option whose type this port does not implement, or "" when all are implemented.
    std::string firstUnsupported() const;
};

// Read system/fvOptions or constant/fvOptions (OpenFOAM looks in both). Absent file => empty list.
OptionList read(const std::string& caseDir, const PrimitiveMesh& m);

// UEqn.H's `== fvOptions(U)`, for a KINEMATIC momentum equation (nu, not mu -- DarcyForchheimer.C
// dispatches on UEqn.dimensions() and takes the `one`/nu branch when the equation is not in force units).
void addSup(
    const OptionList&             opts,
    FvVectorMatrix&               UEqn,
    const GeometricField<vector>& U,
    scalar                        nu,
    const FvGeometry&             g,
    // TRUE for a FORCE-dimensioned momentum equation, which is rhoSimpleFoam's. Both porosity models
    // branch on UEqn.dimensions() in OpenFOAM: fixedCoeff::correct reads `rhoRef` from the dict when the
    // equation is in force units and uses 1 otherwise (fixedCoeff.C:202-207). Passing this rather than
    // inspecting dimensions keeps the _cpp reference free of a dimension set it does not carry.
    bool                          forceDimensions = false);

// fvOptions.constrain(eqn) for a SCALAR equation. `field` is the name the equation solves ("e"/"h" for
// energy, "k", "epsilon", ...); a constraint that does not name it does nothing. The energy equation
// takes fixedTemperatureConstraint, whose value is he(p, Tuniform) -- so the caller supplies the
// conversion rather than this file carrying a thermo.
void constrain(
    const OptionList&           opts,
    FvScalarMatrix&             eqn,
    std::vector<scalar>&        psi,   // setValues writes the constrained value into it, as OF does
    const std::string&          field,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    // he(p, T) for the fixedTemperatureConstraint; unused by the others. Null means an energy constraint
    // is refused rather than applied with a temperature where an energy belongs -- which is exactly the
    // mistake the EEqn gate caught once already.
    scalar                    (*heOfT)(scalar) = nullptr);

} // namespace fvOptions
} // namespace cpu
} // namespace brae
