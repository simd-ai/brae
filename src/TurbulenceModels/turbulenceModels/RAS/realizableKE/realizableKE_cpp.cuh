#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's realizableKE.
//
// provenance:
//   openfoam:
//     file: src/TurbulenceModels/turbulenceModels/RAS/realizableKE/realizableKE.C
//     symbols: realizableKE::correct / rCmu / correctNut
//     keys (ofscan schema realizableKE): A0 4.0, C2 1.9, sigmak 1.0, sigmaEps 1.2
//   brae:
//     reference: src/TurbulenceModels/turbulenceModels/RAS/realizableKE/realizableKE_cpp.cu
//     tests:     tests/test_realizablke_cpp.cu
//
// WHAT MAKES IT "REALIZABLE", and every one of these is a place a port silently becomes standard k-epsilon:
//
//   1. Cmu IS NOT A CONSTANT. nut = rCmu*k^2/eps with
//          S    = devSymm(gradU)
//          W    = 2*sqrt(2)*((S&S)&&S) / (magS*S2 + SMALL)
//          phis = (1/3)*acos(clamp(sqrt(6)*W, -1, 1))
//          As   = sqrt(6)*cos(phis)
//          Us   = sqrt(S2/2 + magSqr(skew(gradU)))
//          rCmu = 1/(A0 + As*Us*k/eps)
//      A port that used a fixed 0.09 here still converges and looks like k-epsilon, because it IS.
//
//   2. S2 = 2*magSqr(devSymm(gradU)) -- devSymm, NOT symm. kOmegaSST's S2 uses symm, and the two differ
//      by the trace term. Same symbol, different quantity, in files a few directories apart.
//
//   3. THE EPSILON EQUATION IS A DIFFERENT EQUATION, not kEpsilon's with new constants:
//          == C1*magS*eps  -  Sp(C2*eps/(k + sqrt(nuLimited*eps)), eps)
//      with C1 = max(eta/(5 + eta), 0.43), eta = magS*k/eps. There is no C1*Cmu*k*GbyNu production term
//      and no divU SuSp term -- both of which kEpsilon's epsilon equation has.
//
// The k equation IS kEpsilon's shape: == G - SuSp((2/3)*divU, k) - Sp(eps/k, k), with G = nut*(gradU &&
// devTwoSymm(gradU)) and DkEff = nut/sigmak + nu.
//
// The WALL FUNCTIONS keep Cmu = 0.09: OpenFOAM's nutkWallFunction/epsilonWallFunction read Cmu from their
// own patch dictionaries and default to 0.09 regardless of the model's variable Cmu.
//
// REFUSED, not ignored: MRF and fvOptions.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include <string>
#include <vector>

namespace brae {

// OF defaults, from ofscan's schema of realizableKE.C. kappa/E/Cmu are the WALL-FUNCTION coefficients
// (OpenFOAM reads them from the BC dictionaries); Cmu stays 0.09 there even though the model's own Cmu
// is variable.
struct RealizableKECoeffs
{
    scalar A0 = 4.0;          // realizableKE.C:151
    scalar C2 = 1.9;          // :160
    scalar sigmak = 1.0;      // :169
    scalar sigmaEps = 1.2;    // :178
    scalar Cmu = 0.09, kappa = 0.41, E = 9.8;   // wall functions only
};

inline void readRealizableKECoeffs(const FoamDict* ras, RealizableKECoeffs& c)
{
    const FoamDict* d = ras ? ras->subDict("realizableKECoeffs") : nullptr;
    if (!d) return;
    c.A0 = d->scalarOr("A0", c.A0);
    c.C2 = d->scalarOr("C2", c.C2);
    c.sigmak = d->scalarOr("sigmak", c.sigmak);
    c.sigmaEps = d->scalarOr("sigmaEps", c.sigmaEps);
    c.kappa = d->scalarOr("kappa", c.kappa);
    c.E = d->scalarOr("E", c.E);
}

namespace cpu {
namespace realizableKE {

// S2 = 2*magSqr(devSymm(gradU)).  devSymm(t) = symm(t) - (1/3)*tr(t)*I
std::vector<scalar> S2(const std::vector<tensor>& gradU);

// The variable Cmu (realizableKE.C::rCmu). Returns rCmu per cell.
std::vector<scalar> rCmu(const std::vector<tensor>& gradU,
                         const std::vector<scalar>& s2,
                         const std::vector<scalar>& k,
                         const std::vector<scalar>& eps,
                         const RealizableKECoeffs& co);

struct RKEResiduals { scalar epsilon = 0, k = 0; };

// One realizableKE::correct(): updates k, epsilon and nut in place.
void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        eps,
    GeometricField<scalar>&        nutField,
    const SurfaceScalarField&      phi,
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxEps,
    scalar                         relaxK,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const RealizableKECoeffs&      co = {},
    RKEResiduals*                  res = nullptr);

} // namespace realizableKE
} // namespace cpu
} // namespace brae
