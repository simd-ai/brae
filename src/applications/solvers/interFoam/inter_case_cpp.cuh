#pragma once
// interFoam's createFields -- the case on disk turned into the fields the solver runs on.
//
// provenance:
//   openfoam:
//     file: applications/solvers/multiphase/interFoam/createFields.H
//     also: applications/solvers/multiphase/VoF/createAlphaFluxes.H
//           src/finiteVolume/cfdTools/general/include/readGravitationalAcceleration.H
//   brae:
//     reference: this header
//     cuda:      (pending)
//     tests:     tests/test_inter_case_cpp.cu, tests/interfoam_createfields_vs_openfoam.sh
//
// WHY THIS FILE IS SEPARATE FROM THE DRIVER, and it is the lesson rhoSimpleFoam's mirror wrote down:
// the harness and the solver must SHARE the case-to-fields translation. A private copy in the driver
// is the defect this project keeps finding one level up -- the gate proves the step, the driver feeds
// it something else, and nothing compares the two. buildInterFields is the one translation, and both
// the gate and brae_interFoam call it.
//
// WHAT createFields.H ACTUALLY BUILDS, in order, and the three places a port goes wrong:
//
//   1. p_rgh and U are READ. alpha1 is read as "alpha." + phase1Name -- damBreak's is alpha.water, and
//      the NAME comes from `phases (water air)` in transportProperties, not from a convention. Reading
//      a hard-coded "alpha1" finds nothing on any shipped case.
//
//   2. alpha2 = 1 - alpha1, then the mixture: rho from the RAW alpha, mu and nu from the CLAMPED one.
//      (two_phase_mixture_cpp.cuh carries that split and its gate.)
//
//   3. phi COMES FROM createAlphaFluxes.H, NOT from fvc::flux(U). OpenFOAM READS `phi` from the start
//      directory if it is there and only computes linearInterpolate(U) & Sf when it is not. A restart
//      therefore continues from the written flux, and recomputing it from U is a different field --
//      U and phi are not consistent to round-off after a solve, and the difference is the continuity
//      error the pressure corrector has just driven down.
//
//   4. gh and ghf need g AND hRef, and ghRef carries g's SIGN (inter_create_fields_cpp.cuh).
//      p = p_rgh + rho*gh is written and never solved.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "time_controls.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "alpha_eqn_cpp.cuh"
#include "inter_ueqn_cpp.cuh"
#include "inter_create_fields_cpp.cuh"
#include "mules_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

struct InterFields
{
    // --- read from the start directory
    GeometricField<scalar> alpha1;
    GeometricField<vector> U;
    GeometricField<scalar> p_rgh;
    SurfaceScalarField     phi;            // read if present, else linearInterpolate(U) & Sf

    // --- derived
    std::vector<scalar> alpha2, rho, mu, nu, gh, p;
    std::vector<scalar> ghfInternal;       // gh on the internal faces
    SurfaceScalarField  rhoPhi;

    // --- the case's own settings
    cpu::twoPhase::MixtureSpec   mixture;
    interfaceProps::InterfaceCoeffs interface;
    AlphaControls                alphaCtl;
    MULES::Controls              mulesCtl;
    VoFTimeControls              timeCtl;
    DivScheme                    divRhoPhiU     = DivScheme::upwind;
    scalar                       divRhoPhiUCoeff = 1.0;
    AlphaFluxScheme              divPhiAlpha    = AlphaFluxScheme::vanLeer;
    AlphaFluxScheme              divPhirbAlpha  = AlphaFluxScheme::linear;
    AlphaDdt                     ddtAlpha       = AlphaDdt::Euler;
    DdtScheme                    ddtU           = DdtScheme::Euler;

    vector  g{0, 0, 0};
    scalar  hRef = 0;
    scalar  ghRefValue = 0;
    scalar  deltaT = 0;
    std::string alphaName;                 // "alpha." + phase1Name
    bool    phiWasRead = false;            // see note 3
};

// The case's dictionaries and fields -> InterFields. Throws, by name, on anything not ported.
InterFields buildInterFields(const std::string&          caseDir,
                             const std::string&          startDir,
                             const PrimitiveMesh&        m,
                             const FvGeometry&           g,
                             const std::vector<FvPatch>& patches);

} // namespace interFoam
} // namespace cpu
} // namespace brae
