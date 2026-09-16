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
    DeviceBuffer<scalar>&        lambdaBnd);

} // namespace brae
