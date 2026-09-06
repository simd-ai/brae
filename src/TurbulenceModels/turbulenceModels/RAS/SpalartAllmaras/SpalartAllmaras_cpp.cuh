#pragma once
// RAS SpalartAllmaras -- the one-equation eddy-viscosity model, host reference.
//
// provenance:
//   openfoam: src/TurbulenceModels/turbulenceModels/Base/SpalartAllmaras/SpalartAllmarasBase.C
//             src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras.C (dTilda = y)
//   brae:     src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras_cpp.cu
//   tests:    tests/test_simple_sa_cpp.cu, tests/sa_cpp_vs_openfoam.sh
//
// One transported scalar, nuTilda, and nut = nuTilda*fv1. The equation (SpalartAllmarasBase.C:459-480):
//
//     div(phi, nuTilda)
//   - laplacian(DnuTildaEff, nuTilda)
//   - Cb2/sigmaNut * magSqr(grad(nuTilda))                 <- EXPLICIT, on the LHS, so source += it
//   ==
//     Cb1*Stilda*nuTilda*(1 - ft2)
//   - Sp((Cw1*fw(Stilda,dTilda) - Cb1/kappa^2*ft2) * nuTilda/dTilda^2, nuTilda)
//
// with DnuTildaEff = (nuTilda + nu)/sigmaNut, dTilda = y (the RAS specialisation), and
// Cw1 = Cb1/kappa^2 + (1 + Cb2)/sigmaNut -- DERIVED, not read, so a case that overrides Cb1/Cb2/kappa
// moves it too.
//
// THREE PLACES THIS MODEL IS EASY TO GET WRONG:
//   * Omega is sqrt(2)*mag(skew(gradU)), the VORTICITY magnitude -- not the strain rate the k-epsilon
//     family uses. skew(t) = (t - t^T)/2, and mag() is the Frobenius norm over all nine components.
//   * r = min(nuTilda/(max(Stilda, SMALL)*sqr(kappa*dTilda)), 10) and its BOUNDARY is set to ZERO
//     (SpalartAllmarasBase.C, `tr.ref().boundaryFieldRef() == 0`). fw is built from r, so a boundary
//     that inherited the cell value would feed the wall destruction term the wrong number.
//   * bound(nuTilda, 0) uses a lower bound of ZERO, not SMALL as k/epsilon/omega do.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "foam_dict.cuh"

#include <vector>

namespace brae {
namespace cpu {
namespace SA {

struct Coeffs
{
    scalar sigmaNut = 0.66666;
    scalar kappa    = 0.41;
    scalar Cb1      = 0.1355;
    scalar Cb2      = 0.622;
    scalar Cw2      = 0.3;
    scalar Cw3      = 2.0;
    scalar Cv1      = 7.1;
    scalar Cs       = 0.3;
    scalar E        = 9.8;      // nutUSpaldingWallFunction: Spalding law E
    scalar nutKappa = 0.41;     // ...and its kappa (the wall function reads its OWN, not the model's)
    // Only reached when the ft2 term is switched on, which OpenFOAM defaults to OFF.
    bool   ft2      = false;
    scalar Ct3      = 1.2;
    scalar Ct4      = 0.5;

    scalar Cw1() const { return Cb1 / (kappa * kappa) + (1.0 + Cb2) / sigmaNut; }
};

void readCoeffs(const FoamDict* ras, Coeffs& c);

// div(phi,nuTilda) scheme. The SA tutorials ask for `bounded Gauss linearUpwind grad(nuTilda)`, which is
// upwind's MATRIX plus a deferred gradient correction -- a different equation from plain upwind.
struct DivScheme
{
    bool bounded       = false;
    bool linearUpwind  = false;
    bool limitedLinear = false;
    scalar coeff       = 1.0;
};

// The INITIAL residual of the nuTilda solve, in OpenFOAM's normalisation.
struct Residuals { scalar nuTilda = 0; };

// One SpalartAllmaras::correct(): updates nuTilda and nut in place.
void correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>&       nuTilda,
    GeometricField<scalar>&       nutField,
    const SurfaceScalarField&     phi,
    const std::vector<scalar>&    y,        // CELL wall distance -- dTilda for the RAS specialisation
    scalar                        nu,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    scalar                        relaxNuTilda,
    scalar                        tol,
    scalar                        relTol,
    int                           maxIter,
    const Coeffs&                 co  = {},
    const DivScheme&              sch = {},
    Residuals*                    res = nullptr);

} // namespace SA
} // namespace cpu
} // namespace brae
