#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's generalizedNewtonian laminar model with the powerLaw
// viscosity.
//
// provenance:
//   openfoam:
//     symbol: Foam::laminarModels::generalizedNewtonian<BasicMomentumTransportModel>
//     file:   src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonian.C
//             :87      nu_ = viscosityModel_->nu(this->nu(), strainRate())   (constructor)
//             :98      strainRate() = sqrt(2)*mag(symm(fvc::grad(this->U())))
//             :128-157 nut() is zero; nuEff() RETURNS nu_, cells and patches
//             :161-165 correct(): nu_ recomputed, then laminarModel::correct()
//     also:   .../generalizedNewtonianViscosityModels/powerLaw/powerLaw.C:62-65 (n, nuMin, nuMax, all
//             mandatory, from optionalSubDict("powerLawCoeffs")) and :96-119 (the formula)
//             src/TurbulenceModels/compressible/CompressibleTurbulenceModel/CompressibleTurbulenceModel.H:123
//             (the compressible nu() is transport.mu()/rho_, rho_ the SOLVER's rho)
//   brae:
//     reference: this file
//     consumers: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu (construction and the
//                step's turbulence->correct())
//     tests:     tests/rho_generalized_newtonian_vs_openfoam.sh
//
// OpenFOAM, verbatim (powerLaw.C:96-119):
//
//     return max
//     (
//         nuMin_,
//         min
//         (
//             nuMax_,
//             nu0*pow
//             (
//                 max
//                 (
//                     dimensionedScalar("one", dimTime, 1)*strainRate,
//                     dimensionedScalar("small", dimless, SMALL)
//                 ),
//                 n_.value() - scalar(1)
//             )
//         )
//     );
//
// THE MODEL REPLACES THE MOLECULAR VISCOSITY, it does not add to it: nuEff() is nu_, and linearViscousStress
// then assembles rho_*nu_ as the momentum diffusivity. On squareBendLiqNoNewtonian nu_ sits at nuMin = 1e-3
// over nearly the whole developed field against a molecular mu/rho of 3.9e-7..9.1e-7.
//
// nu_ IS A STORED FIELD. It is computed once in the constructor and then only in correct(), which
// rhoSimpleFoam calls LAST in the iteration, so iteration n's momentum equation reads the nu_ of the state
// that ended iteration n-1 -- and iteration 1's reads the construction-time one, from the initial U. On a
// case starting from rest that is nuMax nearly everywhere: max(strainRate, SMALL) is SMALL wherever grad(U)
// is exactly zero, and SMALL^(n-1) is enormous.
//
// BOTH HALVES OF THE FIELD. nu_ is a volScalarField built by field algebra, so its patch values are the
// formula applied to the patch values of nu0 and strainRate -- and strainRate's patch values come from
// grad(U)'s, which gaussGrad corrects so the normal component is the patch's own snGrad
// (gaussGrad::correctBoundaryConditions). On a no-slip wall that is the wall shear, and rho_b*nu_b is the
// coefficient the wall face of the momentum laplacian takes.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include <cmath>
#include <vector>

namespace brae {
namespace cpu {
namespace generalizedNewtonian {

// powerLaw.C:62-65 -- all three constructed from the dictionary with no default.
struct PowerLawCoeffs
{
    scalar n     = 1.0;
    scalar nuMin = 0.0;
    scalar nuMax = 0.0;
};

// powerLaw::nu for one value. BRAE_HD so the device arm evaluates the same line. SMALL is OpenFOAM's
// double-precision 1e-15 (doubleScalar.H); `one` [s] only carries the dimensions.
BRAE_HD inline scalar powerLawNu(
    scalar nu0,
    scalar strainRate,
    scalar n,
    scalar nuMin,
    scalar nuMax)
{
    const scalar small = scalar(1e-15);
    const scalar sr    = strainRate > small ? strainRate : small;
    const scalar nu    = nu0 * std::pow(sr, n - scalar(1));
    const scalar upper = nuMax < nu ? nuMax : nu;
    return nuMin > upper ? nuMin : upper;
}

// strainRate() = sqrt(2)*mag(symm(gradU)) for one gradient tensor.
BRAE_HD inline scalar strainRate(const tensor& gradU)
{
    return std::sqrt(scalar(2)) * std::sqrt(magSqr(symm(gradU)));
}

// nu_ = viscosityModel_->nu(this->nu(), strainRate()), cells and every patch face.
//   U            the velocity as it stands (boundary values current -- the same field fvc::grad reads)
//   nu0/nu0Bnd   this->nu() = mu/rho_, per cell and per boundary face
//   gradULimitK  the cellLimited coefficient of gradSchemes/grad(U) (0 = Gauss linear, unlimited);
//                fvc::grad(U) resolves THAT entry, named-then-default
void correctNu(
    const GeometricField<vector>&           U,
    const std::vector<scalar>&              nu0,
    const std::vector<std::vector<scalar>>& nu0Bnd,
    const PowerLawCoeffs&                   coeffs,
    scalar                                  gradULimitK,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches,
    std::vector<scalar>&                    nu,
    std::vector<std::vector<scalar>>&       nuBnd);

} // namespace generalizedNewtonian
} // namespace cpu
} // namespace brae
