#pragma once
// MULES -- the Multidimensional Universal Limiter for Explicit Solution. Host reference.
//
// provenance:
//   openfoam:
//     file: src/finiteVolume/fvMatrices/solvers/MULES/MULESTemplates.C
//             :195-270   limiter()  -- the controls and the neighbourhood extrema
//             :271-390   the face scan: sumPhiBD, sumPhip, mSumPhim, and the boundary branches
//             :391-437   the bounds converted into flux-space budgets
//             :438-568   THE ITERATION
//             :571-635   limit()    -- phiBD, phiCorr, and phiPsi = phiBD + lambda*phiCorr
//             :20-70     explicitSolve() -- the update the limiter is protecting
//   brae:
//     reference: this header
//     cuda:      src/finiteVolume/fvMatrices/solvers/MULES/device_mules.cu  (not written yet)
//     tests:     tests/test_mules_cpp.cu
//
// WHAT MULES IS, because it is unlike everything else in this tree. Every other numerical module here
// assembles a matrix. MULES does not. It takes two face fluxes -- a first-order UPWIND one that is
// guaranteed bounded, and the high-order one the case actually asked for -- and finds a per-face factor
// lambda in [0,1] blending between them:
//
//     phiBD   = upwind(phi).flux(psi)            the bounded donor
//     phiCorr = phiPsi - phiBD                   the antidiffusive correction
//     phiPsi  = phiBD + lambda*phiCorr           what gets used
//
// lambda = 1 everywhere means "the high-order flux was already safe"; lambda = 0 on a face means "fall
// back to upwind there". The whole algorithm is the search for the largest lambda that still keeps
// every cell inside [psiMin, psiMax]. It is explicit, iterative and bound-preserving, and it is what
// makes a VoF interface stay an interface instead of smearing or going out of bounds.
//
// SIX THINGS A TRANSCRIPTION OF THIS GETS WRONG, none of which shows in a converged field:
//
// 1. psiMaxn AND psiMinn ARE INITIALISED SWAPPED (MULESTemplates.C:280-281):
//        psiMaxn = psiMin;     psiMinn = psiMax;
//    They start at the OPPOSITE bound so that the running max/min over the neighbours builds up from
//    the far end. Initialising them "naturally" -- psiMaxn = psiMax -- makes the neighbourhood scan a
//    no-op, every cell gets the global bounds, and the limiter stops limiting. The field then goes
//    unbounded only where the flux is strong enough, which on a dam break is after several hundred
//    steps.
//
// 2. THE NEIGHBOURHOOD EXTREMA EXCLUDE THE CELL'S OWN VALUE. The scan takes psi[nei] into psiMaxn[own]
//    and psi[own] into psiMaxn[nei] -- never psi[own] into psiMaxn[own]. A cell is allowed to move to
//    its neighbours' extremes, which is what lets an interface advance at all.
//
// 3. phiBD IS OVERWRITTEN WITH phiPsi ON EVERY NON-COUPLED BOUNDARY (MULESTemplates.C:600-609), which
//    makes phiCorr identically zero there. The boundary flux is NEVER limited: whatever the boundary
//    condition prescribed is what crosses. Limiting it instead would silently reduce a prescribed
//    inflow, which reads as a boundary-condition bug a long way from here.
//
// 4. A WEDGE PATCH GETS lambda = 0 (MULESTemplates.C:530-533), unconditionally, before any of the
//    coupled logic. 2D axisymmetric cases are wedges, and this is the only place the geometry enters.
//
// 5. LAMBDA ONLY EVER DECREASES, so nLimiterIter is a monotone tightening rather than a convergence
//    loop -- more iterations can only limit more. The reason is not the running `min(lambda, ...)`:
//    sumlPhip and mSumlPhim are built FROM lambda, so they shrink with it and each pass produces a
//    value already below the previous one. Replacing the min with a plain assignment was measured here
//    and gave a bit-identical field. It is kept because OpenFOAM keeps it, not because it is doing
//    work. What DOES depend on nLimiterIter is how far the tightening propagates between cells whose
//    faces constrain each other -- which needs more than one dimension to happen at all.
//
// 6. THE DIVISORS CARRY ROOTVSMALL, NOT VSMALL AND NOT A GUARD. mSumPhim and sumPhip are zero in every
//    cell the correction does not touch -- which is most of the domain -- so this divide happens
//    everywhere and the constant chosen decides what lambda is in the untouched cells.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace MULES {

// mesh.solverDict(psi.name()) -- the SAME fvSolution entry alphaControls reads, so on damBreak this is
// `"alpha.water.*"` and nLimiterIter is 5 there, not the default 3.
struct Controls
{
    label  nLimiterIter         = 3;     // getOrDefault, MULESTemplates.C:205
    scalar smoothLimiter        = 0;     // :210
    scalar extremaCoeff         = 0;     // :215
    scalar boundaryExtremaCoeff = 0;     // :220 -- DEFAULTS TO extremaCoeff, not to 0
};

Controls readControls(const FoamDict& fvSolution, const std::string& psiName);

// The optional fields. Every one of these is a template parameter in OpenFOAM, instantiated for
// interFoam as geometricOneField / zeroField / oneField, and a null pointer here means exactly that
// constant. They are pointers rather than defaulted vectors so that "interFoam passes one" and "a
// solver with phase change passes a field" are visibly different at the call site.
struct Fields
{
    const std::vector<scalar>* rho    = nullptr;   // null == geometricOneField()   (interFoam)
    const std::vector<scalar>* rhoOld = nullptr;   // null == geometricOneField()
    const std::vector<scalar>* Sp     = nullptr;   // null == zeroField()           (interFoam)
    const std::vector<scalar>* Su     = nullptr;   // null == zeroField()
    const std::vector<scalar>* psiMax = nullptr;   // null == oneField()            (interFoam)
    const std::vector<scalar>* psiMin = nullptr;   // null == zeroField()
};

// lambda, on every face of the mesh. Kept as one object because OpenFOAM's allLambda is one array
// sliced into internal and boundary halves, and the iteration writes both.
struct Limiter
{
    std::vector<scalar>              internal;   // nInternalFaces
    std::vector<std::vector<scalar>> boundary;   // [patch][face]
};

// MULES::limiter, MULESTemplates.C:195-568. phiBD and phiCorr are supplied already formed, because
// limit() below is where they are built and the two steps are separately wrong in different ways.
void limiter(Limiter&                      lambda,
             scalar                        rDeltaT,
             const GeometricField<scalar>& psi,
             const std::vector<scalar>&    psiOld,
             const SurfaceScalarField&     phiBD,
             const SurfaceScalarField&     phiCorr,
             const Fields&                 f,
             const Controls&               c,
             const PrimitiveMesh&          m,
             const FvGeometry&             g,
             const std::vector<FvPatch>&   patches);

// The bounded donor flux, upwind(phi).flux(psi), with the boundary overwritten by phiPsi on every
// non-coupled patch -- see note 3. Exposed because that overwrite is the step most likely to be left
// out, and a gate needs to be able to look at it on its own.
void boundedDonorFlux(const SurfaceScalarField&     phi,
                      const GeometricField<scalar>& psi,
                      const SurfaceScalarField&     phiPsi,
                      const PrimitiveMesh&          m,
                      const std::vector<FvPatch>&   patches,
                      SurfaceScalarField&           phiBD);

// MULES::limit, MULESTemplates.C:571-635. phiPsi is REWRITTEN in place as phiBD + lambda*phiCorr.
// `lambdaOut`, when non-null, receives the limiter -- a stock OpenFOAM run never writes it, and that
// is the whole reason this signature exists.
void limit(scalar                        rDeltaT,
           const GeometricField<scalar>& psi,
           const std::vector<scalar>&    psiOld,
           const SurfaceScalarField&     phi,
           SurfaceScalarField&           phiPsi,
           const Fields&                 f,
           const Controls&               c,
           const PrimitiveMesh&          m,
           const FvGeometry&             g,
           const std::vector<FvPatch>&   patches,
           Limiter*                      lambdaOut = nullptr);

// MULES::explicitSolve, MULESTemplates.C:20-70 -- the update the limiter exists to protect, on a fixed
// mesh (Vsc == Vsc0):
//
//     psi = (rho.oldTime()*psi0*rDeltaT + Su - surfaceIntegrate(phiPsi)) / (rho*rDeltaT - Sp)
//
// NOTE rho.oldTime() IN THE NUMERATOR AND rho IN THE DENOMINATOR, the same split fvm::ddt(rho,U) has.
void explicitSolve(scalar                      rDeltaT,
                   std::vector<scalar>&        psi,
                   const std::vector<scalar>&  psiOld,
                   const SurfaceScalarField&   phiPsi,
                   const Fields&               f,
                   const PrimitiveMesh&        m,
                   const FvGeometry&           g,
                   const std::vector<FvPatch>& patches);

// The whole thing: limit, then solve. This is what alphaEqn.H:210-220 calls.
void explicitSolveLimited(scalar                        rDeltaT,
                          GeometricField<scalar>&       psi,
                          const std::vector<scalar>&    psiOld,
                          const SurfaceScalarField&     phi,
                          SurfaceScalarField&           phiPsi,
                          const Fields&                 f,
                          const Controls&               c,
                          const PrimitiveMesh&          m,
                          const FvGeometry&             g,
                          const std::vector<FvPatch>&   patches,
                          Limiter*                      lambdaOut = nullptr);

// ---------------------------------------------------------------------------------------------------
// CMULES -- the SEMI-IMPLICIT path, selected by `MULESCorr yes` (13 of the 44 shipped interFoam
// tutorials, damBreak among them).
//
//   provenance: src/finiteVolume/fvMatrices/solvers/MULES/CMULESTemplates.C
//                 :38-76    correct()     -- the update
//                 :203-568  limiterCorr() -- the limiter
//                 :570-627  limitCorr()   -- phiCorr *= lambda
//
// WHAT CHANGES, AND WHY. With MULESCorr the upwind part of the alpha equation is solved IMPLICITLY as a
// matrix (alphaEqn.H:103-122) instead of being carried explicitly. So by the time CMULES runs, psi has
// ALREADY been advanced a full time step; what is left is the antidiffusive correction alone. That one
// fact produces every difference below, and each is a place a port that reuses the explicit code is
// quietly wrong.
//
// A. correct() USES THE CURRENT psi AND THE CURRENT rho:
//
//        psi = (rho*psi*rDeltaT + Su - surfaceIntegrate(phiCorr)) / (rho*rDeltaT - Sp)
//
//    where explicitSolve uses rho.oldTime()*psi.oldTime(). Substituting psi.oldTime() here throws away
//    the implicit solve's result and re-does the time step from the old state with only the correction
//    flux -- which loses the entire upwind advection. The field stays bounded, so a boundedness gate
//    does not notice; the interface simply stops moving at the right speed.
//
// B. THE BUDGET TRANSFORM CARRIES NO sumPhiBD (CMULESTemplates.C:400-412). There is no bounded donor
//    flux in CMULES -- it went through the matrix -- so the budget is measured against psi as it
//    stands, not against psi.oldTime() plus a donor step. Copying the explicit transform brings a term
//    that does not exist here.
//
// C. UNCOUPLED BOUNDARY FACES ARE LIMITED, BUT OUTLETS ONLY (CMULESTemplates.C:537-561):
//
//        if ((phi[f] + phiCorr[f]) > SMALL*SMALL)   // "Limit outlet faces only"
//
//    In explicit MULES the boundary correction is identically zero (phiBD is overwritten with phiPsi),
//    so its uncoupled patches need no branch at all. Here phiCorr arrives from the caller and IS
//    non-zero on the boundary, so it must be limited -- but only where the total flux leaves the
//    domain. Limiting an inlet would throttle a prescribed inflow.
//
// D. nLimiterIter IS MANDATORY HERE (`get<label>`, CMULESTemplates.C:225), where the explicit limiter
//    defaults it to 3. A case that sets `MULESCorr yes` and omits nLimiterIter is a FatalError in
//    OpenFOAM. Defaulting it would run the case OpenFOAM refuses, with an iteration count nobody chose.
//
// The two limiters are kept as two functions, as OpenFOAM keeps them, rather than merged behind a flag.
// Merging would put A, B and C behind branches in one body and hide precisely what this comment exists
// to name.

// Like readControls, but nLimiterIter has NO DEFAULT -- see D.
Controls readControlsCorr(const FoamDict& fvSolution, const std::string& psiName);

// MULES::limiterCorr, CMULESTemplates.C:203-568. `phi` is needed only for the outlet test in C.
void limiterCorr(Limiter&                      lambda,
                 scalar                        rDeltaT,
                 const GeometricField<scalar>& psi,
                 const SurfaceScalarField&     phi,
                 const SurfaceScalarField&     phiCorr,
                 const Fields&                 f,
                 const Controls&               c,
                 const PrimitiveMesh&          m,
                 const FvGeometry&             g,
                 const std::vector<FvPatch>&   patches);

// MULES::limitCorr, CMULESTemplates.C:570-627: phiCorr *= lambda, IN PLACE. Note it does not rebuild a
// blended flux -- there is nothing to blend against.
void limitCorr(scalar                        rDeltaT,
               const GeometricField<scalar>& psi,
               const SurfaceScalarField&     phi,
               SurfaceScalarField&           phiCorr,
               const Fields&                 f,
               const Controls&               c,
               const PrimitiveMesh&          m,
               const FvGeometry&             g,
               const std::vector<FvPatch>&   patches,
               Limiter*                      lambdaOut = nullptr);

// MULES::correct, CMULESTemplates.C:38-76 -- see A.
void correct(scalar                      rDeltaT,
             std::vector<scalar>&        psi,
             const SurfaceScalarField&   phiCorr,
             const Fields&               f,
             const PrimitiveMesh&        m,
             const FvGeometry&           g,
             const std::vector<FvPatch>& patches);

// limitCorr then correct -- what alphaEqn.H:183-193 calls.
void correctLimited(scalar                        rDeltaT,
                    GeometricField<scalar>&       psi,
                    const SurfaceScalarField&     phi,
                    SurfaceScalarField&           phiCorr,
                    const Fields&                 f,
                    const Controls&               c,
                    const PrimitiveMesh&          m,
                    const FvGeometry&             g,
                    const std::vector<FvPatch>&   patches,
                    Limiter*                      lambdaOut = nullptr);

} // namespace MULES
} // namespace cpu
} // namespace brae
