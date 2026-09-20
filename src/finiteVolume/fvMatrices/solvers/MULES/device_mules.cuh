#pragma once
// MULES on the device -- the explicit bound-preserving limiter.
//
// provenance:
//   openfoam:  src/finiteVolume/fvMatrices/solvers/MULES/MULESTemplates.C
//   host:      src/finiteVolume/fvMatrices/solvers/MULES/mules_cpp.cu -- the ORACLE for this file,
//              and itself gated on boundedness against an unlimited control.
//   tests:     tests/test_device_mules.cu
//
// NO ATOMICS. Every one of MULES' five face-to-cell accumulations -- the neighbourhood extrema,
// sumPhiBD, sumPhip, mSumPhim, and sumlPhip/mSumlPhim once per iteration -- is a GATHER over the
// mesh's own owner/losort addressing, the same walk deviceDiv and deviceGaussGrad use. That is not
// only faster than atomicAdd on doubles; it makes the summation order DETERMINISTIC, so two runs of
// the same case give the same limiter. A limiter whose answer depends on scheduling would make every
// boundedness failure unreproducible, which is the one thing this component must not be.
//
// WHAT CANNOT BE BIT-IDENTICAL TO THE HOST, and why the gate is what it is. Each cell's sums run in a
// different ORDER from the host's face loop, and floating-point addition is not associative -- so the
// budgets, and therefore lambda, differ in the last bits. That difference then passes through a
// clamp01 and a min, which mostly absorbs it but can flip a face that sits exactly on a bound. The
// gate therefore holds the DEVICE to the same PROPERTY the host is held to -- alpha stays in [0,1],
// exactly -- and compares lambda within a measured tolerance rather than bit for bit.
//
// THE PATCH FLAGS. `bndFlag` is one int per boundary face:
//     0  ordinary          -- in every sum, lambda free
//     1  empty             -- in NO sum, as emptyFvPatch::size() == 0 means in OpenFOAM
//     2  wedge             -- in every sum, but lambda forced to 0 (MULESTemplates.C:530-533)
// They are passed rather than derived because the device mesh does not carry a patch type, and
// inventing one for two special cases would put a fact about boundary conditions in the geometry.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"

namespace brae {

struct DeviceMulesControls
{
    label  nLimiterIter         = 3;
    scalar smoothLimiter        = 0;
    scalar extremaCoeff         = 0;
    scalar boundaryExtremaCoeff = 0;
};

// Bounds and sources. Null means the constant OpenFOAM instantiates for interFoam --
// geometricOneField for rho, zeroField for Sp/Su, oneField/zeroField for psiMax/psiMin.
struct DeviceMulesFields
{
    const scalar* rho    = nullptr;   // null == 1
    const scalar* rhoOld = nullptr;   // null == 1
    const scalar* Sp     = nullptr;   // null == 0
    const scalar* Su     = nullptr;   // null == 0
    const scalar* psiMax = nullptr;   // null == 1
    const scalar* psiMin = nullptr;   // null == 0
};

// phiBD = upwind(phi).flux(psi), with the boundary OVERWRITTEN by phiPsi on every non-coupled patch --
// so phiCorr is identically zero there and the prescribed boundary flux is never limited
// (MULESTemplates.C:600-609).
void deviceMulesDonorFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    int                         nBoundaryFaces,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& phiPsiBnd,
    DeviceBuffer<scalar>&       phiBDInt,
    DeviceBuffer<scalar>&       phiBDBnd);

// MULES::limiter. `lambda` is sized and filled by this call: 1 on every face, then tightened.
void deviceMulesLimiter(
    const DeviceMesh&            dm,
    int                          nInternalFaces,
    int                          nBoundaryFaces,
    scalar                       rDeltaT,
    const DeviceBuffer<scalar>&  psi,
    const DeviceBuffer<scalar>&  psiOld,
    const DeviceBuffer<scalar>&  psiBndValue,     // psi's patch values, for the fixesValue extrema
    const DeviceBuffer<int>&     bndFixesValue,   // 1 where the patch fixes a value
    const DeviceBuffer<int>&     bndFlag,         // 0 ordinary, 1 empty, 2 wedge
    const DeviceBuffer<scalar>&  phiBDInt,
    const DeviceBuffer<scalar>&  phiBDBnd,
    const DeviceBuffer<scalar>&  phiCorrInt,
    const DeviceBuffer<scalar>&  phiCorrBnd,
    const DeviceMulesFields&     f,
    const DeviceMulesControls&   c,
    DeviceBuffer<scalar>&        lambdaInt,
    DeviceBuffer<scalar>&        lambdaBnd,
    // THE COUPLED FACES. A cyclic patch is not in the boundary arrays at all (device_mesh.cuh:41-44),
    // so without these MULES limits a mesh with a periodic pair as if the pair were a wall: its faces
    // contribute no extrema, no budget and no limiter. All four together or none.
    const DeviceCyclic*          cyc = nullptr,
    const DeviceBuffer<scalar>*  phiBDIf = nullptr,
    const DeviceBuffer<scalar>*  phiCorrIf = nullptr,
    DeviceBuffer<scalar>*        lambdaIf = nullptr);

// phiPsi = phiBD + lambda*phiCorr (MULESTemplates.C:634). The blend is separate from the limiter so a
// caller can keep the limiter's lambda -- which a stock OpenFOAM run never writes and which every gate
// on this component needs.
void deviceMulesBlend(
    int                         nInternalFaces,
    int                         nBoundaryFaces,
    const DeviceBuffer<scalar>& phiBDInt,  const DeviceBuffer<scalar>& phiBDBnd,
    const DeviceBuffer<scalar>& lambdaInt, const DeviceBuffer<scalar>& lambdaBnd,
    const DeviceBuffer<scalar>& phiCorrInt,const DeviceBuffer<scalar>& phiCorrBnd,
    DeviceBuffer<scalar>&       phiPsiInt, DeviceBuffer<scalar>&       phiPsiBnd);

// MULES::explicitSolve on a fixed mesh (MULESTemplates.C:20-70):
//     psi = (rho.oldTime()*psi0*rDeltaT + Su - surfaceIntegrate(phiPsi)) / (rho*rDeltaT - Sp)
// NOTE rho.oldTime() above the line and rho below it -- the same split fvm::ddt(rho,U) carries, and on
// a VoF interface those differ by the density ratio.
void deviceMulesExplicitSolve(
    const DeviceMesh&           dm,
    scalar                      rDeltaT,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& phiPsiInt,
    const DeviceBuffer<scalar>& phiPsiBnd,
    const DeviceMulesFields&    f,
    DeviceBuffer<scalar>&       psi);

// ---------------------------------------------------------------------------------------------------
// CMULES on the device -- the SEMI-IMPLICIT path, `MULESCorr yes`, which damBreak and twelve other
// shipped tutorials select.
//
//   provenance: src/finiteVolume/fvMatrices/solvers/MULES/CMULESTemplates.C
//   host:       mules_cpp.cu, limiterCorr / limitCorr / correct -- the ORACLE, whose own header spells
//               out A, B, C and D, the four things that make this not the explicit path.
//
// Three of those four are visible in the signatures below and are worth naming here too, because the
// temptation on the device is to reach for the explicit kernels with a flag:
//
//   A. correct() takes the CURRENT psi and the CURRENT rho, where the explicit solve takes
//      psi.oldTime() and rho.oldTime(). By the time CMULES runs, the implicit upwind matrix has
//      ALREADY advanced psi a whole time step; substituting the old values throws that away and
//      re-does the step carrying only the correction. The field stays bounded, so a boundedness gate
//      does not notice -- the interface simply moves at the wrong speed.
//
//   B. There is NO phiBD and NO sumPhiBD anywhere below. The donor flux went through the matrix, so
//      the budget is measured against psi as it stands. deviceMulesLimiterCorr therefore takes no
//      phiBD argument at all, rather than one it would have to be passed zero.
//
//   C. Uncoupled boundary faces ARE limited here -- phiCorr arrives from the caller and is genuinely
//      non-zero on them -- but OUTLETS ONLY, tested on `phi + phiCorr`, the TOTAL flux, against
//      SMALL*SMALL rather than zero. That is why `phiBnd` appears in the limiter's arguments when the
//      explicit one needs no such thing.
//
// The same no-atomics gather structure as the explicit path. The consequence differs only in degree:
// bit-identity is not guaranteed, because the budgets sum in a different order from the host's face
// loop -- but MEASURED on the gate's fixture, CMULES' lambda comes out exact on all 264 faces (32 of
// them strictly inside (0,1)), where the explicit limiter differs on five by 3.7e-15. The gate keeps a
// tolerance rather than asserting equality, because what was measured on one fixture is not a
// guarantee, and asserts boundedness exactly.

// MULES::limiterCorr. `lambdaInt`/`lambdaBnd` are sized and filled here: 1 on every face, then
// tightened nLimiterIter times.
void deviceMulesLimiterCorr(
    const DeviceMesh&            dm,
    int                          nInternalFaces,
    int                          nBoundaryFaces,
    scalar                       rDeltaT,
    const DeviceBuffer<scalar>&  psi,             // CURRENT, post-implicit-solve -- see A
    const DeviceBuffer<scalar>&  psiBndValue,
    const DeviceBuffer<int>&     bndFixesValue,
    const DeviceBuffer<int>&     bndFlag,         // 0 ordinary, 1 empty, 2 wedge
    const DeviceBuffer<scalar>&  phiBnd,          // the boundary VOLUMETRIC flux -- see C
    const DeviceBuffer<scalar>&  phiCorrInt,
    const DeviceBuffer<scalar>&  phiCorrBnd,
    const DeviceMulesFields&     f,
    const DeviceMulesControls&   c,
    DeviceBuffer<scalar>&        lambdaInt,
    DeviceBuffer<scalar>&        lambdaBnd);

// MULES::limitCorr: the limiter, then phiCorr *= lambda IN PLACE. No blended flux is formed -- there is
// nothing to blend against, which is the shape difference B leaves behind.
void deviceMulesLimitCorr(
    const DeviceMesh&            dm,
    int                          nInternalFaces,
    int                          nBoundaryFaces,
    scalar                       rDeltaT,
    const DeviceBuffer<scalar>&  psi,
    const DeviceBuffer<scalar>&  psiBndValue,
    const DeviceBuffer<int>&     bndFixesValue,
    const DeviceBuffer<int>&     bndFlag,
    const DeviceBuffer<scalar>&  phiBnd,
    DeviceBuffer<scalar>&        phiCorrInt,      // scaled in place
    DeviceBuffer<scalar>&        phiCorrBnd,      // scaled in place
    const DeviceMulesFields&     f,
    const DeviceMulesControls&   c,
    DeviceBuffer<scalar>*        lambdaIntOut = nullptr,
    DeviceBuffer<scalar>*        lambdaBndOut = nullptr);

// MULES::correct, CMULESTemplates.C:38-76:
//     psi = (rho*psi*rDeltaT + Su - surfaceIntegrate(phiCorr)) / (rho*rDeltaT - Sp)
// Both rho and psi on the right are the CURRENT ones -- see A.
void deviceMulesCorrect(
    const DeviceMesh&           dm,
    scalar                      rDeltaT,
    const DeviceBuffer<scalar>& phiCorrInt,
    const DeviceBuffer<scalar>& phiCorrBnd,
    const DeviceMulesFields&    f,
    DeviceBuffer<scalar>&       psi);            // in and out

} // namespace brae
