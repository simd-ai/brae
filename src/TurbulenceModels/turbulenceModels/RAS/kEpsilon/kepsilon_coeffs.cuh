#pragma once
// k-epsilon model coefficients, shared by the device (device_kepsilon) and CPU (k_epsilon/parallel_kepsilon)
// paths. Defaults = OpenFOAM v2412 kEpsilon defaults; read from turbulenceProperties RAS.kEpsilonCoeffs
// (Cmu/C1/C2/C3/sigmak/sigmaEps). kappa/E are the wall-function coeffs (OF reads them from the wall-function
// BC dicts; default 0.41/9.8). A default-constructed struct reproduces the previous hardcoded constants exactly,
// so unchanged callers stay bit-identical.
#include "cf_types.cuh"

namespace brae {

struct KEpsilonCoeffs
{
    scalar Cmu = 0.09, C1 = 1.44, C2 = 1.92, C3 = 0.0;
    // Foam::bound's lower bounds, from the TOP LEVEL of the case's RAS (or LES) sub-dict, not from a
    // ...Coeffs one: RASModel.C:73-99 getOrAddToDict("kMin", RASDict_, ..., SMALL), re-read at :180-182,
    // and LESModel.C:82-111 carries the same three keys for the DES arms. They are not coefficients of
    // this model -- they live on the base class -- but they ride here because this struct is what
    // already reaches every bound site on both arms (epsLowRe below is the same precedent), and brae
    // hardcoded 1e-15 at all eighteen of them. SMALL is 1e-15 in double precision.
    scalar kMin = 1e-15, epsilonMin = 1e-15;
    scalar sigmaK = 1.0, sigmaEps = 1.3;
    scalar kappa = 0.41, E = 9.8;
    // THE WALL FUNCTIONS' OWN Cmu. OpenFOAM's nutkWallFunction, epsilonWallFunction and
    // omegaWallFunction all read wallCoeffs_.Cmu() -- the nut wall function patch's coefficient, default
    // 0.09 -- and never the model's (nutkWallFunctionFvPatchScalarField.C:43, epsilonWallFunction...C:192,
    // omegaWallFunction...C:191). brae handed them the MODEL's Cmu, which is the same number at the
    // defaults and a different one the moment a case changes it: rhoSST with betaStar 0.1 parted from
    // OpenFOAM at iteration 1 by omega 2.5e-02, all of it in the wall rows.
    scalar CmuWall = 0.09;
    // The case's laplacianSchemes, for the k and epsilon diffusion terms. `Gauss linear corrected` changes
    // the implicit face coefficient AND adds an explicit non-orthogonal source; kOmegaSST already takes
    // both. FALSE here is `orthogonal`/`uncorrected`, so a caller that does not set it keeps the
    // arithmetic it had -- but a case whose fvSchemes says `corrected` must set it, or the closure is
    // solving a different discretisation from the one the case asked for.
    // The case's gradSchemes coefficients: `grad(U) cellLimited Gauss linear <k>` for fvc::grad(U) in the
    // production (kEpsilon.C:237) and in the momentum laplacian's correction, `grad(k)`/`grad(epsilon)` for
    // the k and epsilon laplacians' non-orthogonal corrections (correctedSnGrad.C:52-55 takes each field's
    // OWN grad scheme). 0 = unlimited. Measured on naca0012 (`limited cellLimited Gauss linear 1` on all
    // three) with these unlimited: k 3.3e-04, omega 5.4e-03, nut 1.2e-03 against OpenFOAM at t = 1.
    scalar gradULimitK = 0.0;
    // grad(U)'s BASE scheme, when the case's gradSchemes resolve it to leastSquares rather than Gauss
    // linear (the cellLimited coefficient above applies on top of either). fvc::grad(U) is taken by the
    // closure's production (kOmegaSSTBase.C:522, kEpsilon.C:237) and correctNut; a `default leastSquares`
    // reaches it, and the shared parser used to WARN and run Gauss (scheme_parse.cuh: "approximated as
    // Gauss linear"). Measured on validation/rhoSST restarted from OpenFOAM's iteration 5 under
    // `default leastSquares` with everything else exact: U 3.1e-05, k 2.4e-04, nut 1.3e-03 at iteration 6.
    bool   gradULeastSq = false;
    scalar gradKLimitK = 0.0;
    // The SCHEME those two entries name, when it is not Gauss: `leastSquares` (OpenFOAM's inverse-distance
    // least-squares fit, leastSquaresGrad.C), which the case gasMixing/injectorPipe and validation/rhoSST's
    // leastSquares variant set as `default`. Both fields carry one flag, as they carry one cellLimited
    // coefficient; a cellLimited <k> on top of it still applies, as cellLimitedGrad wraps any base scheme.
    bool   gradKLeastSq = false;
    bool   correctedLaplacian = false;
    scalar snGradLimitCoeff   = 0.0;    // `limited <k> corrected`; 0 = unlimited
    // realizableKE (OF RAS/realizableKE): variable Cmu (rCmu from strain invariants), strain-based eps production
    // C1=max(eta/(5+eta),0.43)*magS*eps, destruction C2*eps^2/(k+sqrt(nu*eps)). Defaults A0=4, C2=1.9, sigmaEps=1.2.
    bool   realizable = false;
    scalar A0 = 4.0;
    // RNGkEpsilon (OF RAS/RNGkEpsilon): standard k-epsilon with renormalisation-group coefficients and ONE
    // extra term -- the epsilon production coefficient becomes (C1 - R) instead of C1, with
    //     eta = sqrt(|S2|)*k/epsilon,   R = eta*(1 - eta/eta0)/(1 + beta*eta^3)
    // R is a strain-rate-dependent SINK at high strain (eta > eta0) and a source below it, which is what
    // lets RNG handle separated and swirling flow that the standard model over-predicts. The SuSp divU
    // term keeps the plain C1 -- OF applies (C1 - R) to the G production alone.
    bool   rng  = false;
    scalar eta0 = 4.38;
    scalar beta = 0.012;
    // epsilonWallFunction `lowReCorrection` (epsilonWallFunctionFvPatchScalarField.C:414,
    // getOrDefault("lowReCorrection", false)) -- a property of the epsilon WALL BC rather than of the
    // model, carried here because this struct already reaches both the _cpp reference and every device
    // entry point that needs it. On a wall face with y+ < yPlusLam it takes epsilon = 2*k*nu/y^2 in place
    // of the log form AND contributes no wall production at all (:242 and :338). Both halves or neither:
    // the log epsilon under-predicts dissipation on a resolved mesh while the production keeps feeding k.
    bool   epsLowRe = false;
};

} // namespace brae
