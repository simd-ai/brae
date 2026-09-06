#pragma once
// ---------------------------------------------------------------------------------------------------------------
// Unified coupled-interface host API (cyclic + cyclicAMI).
//
// WHY: the SIMPLE solver, the turbulence transport, and divDevReff each apply the SAME conceptual operation to
// whatever coupled interfaces the mesh carries -- "add the interface's gaussGrad contribution", "assemble the
// interface Laplacian off-diagonal", "build the interface flux". Historically each call site spelled this out
// twice, once as deviceCyclic<Op>(cyc_, ...) and once as deviceAmi<Op>(ami_, ...), with the two argument lists
// kept manually in lock-step. This header collapses that duplication into a single overloaded free function per
// operation: interface<Op>(cyc_, ...) and interface<Op>(ami_, ...) resolve by the interface object's type to the
// existing kernel wrapper. No numerics change -- these are inline forwarders; overload resolution is the whole
// mechanism. A future processor/GDSW interface adds a third overload here and every call site picks it up for free.
//
// SCOPE: the 15 operations below have IDENTICAL argument lists after the interface object in both backends, so they
// unify cleanly. Operations that are intentionally NOT unified here (call the backend function directly):
//   * cyclic-only : deviceCyclicAddConvection, deviceCyclicFluxRot, deviceCyclicAddHRot, deviceCyclicAddHDiag
//   * AMI-only    : deviceAmiInterpolate, deviceAmiInterpolateVec, deviceAmiAmul, deviceAmiAmulCoeff
//   * AddGradRot  : the two backends take genuinely different arguments -- cyclic rotates on the fly from the three
//                   velocity components (Ux,Uy,Uz,comp,V,...), AMI consumes a precomputed own + rotated-neighbour
//                   pair (Uown,UNbr,V,...) -- so there is no common signature to overload. Kept backend-specific.
//
// Processor/parallel interfaces are deliberately absent (multi-GPU path is on standby); adding them is purely
// additive -- new overloads, no change to existing ones.
// ---------------------------------------------------------------------------------------------------------------
#include "device_cyclic.cuh"
#include "device_ami.cuh"

namespace brae {

// --- matrix assembly ------------------------------------------------------------------------------------------
inline void interfaceAssembleLaplacian(DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                                       DeviceBuffer<scalar>& diag, bool addToDiag = true)
{ deviceCyclicAssembleLaplacian(cyc, gammaCell, diag, addToDiag); }
inline void interfaceAssembleLaplacian(DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell,
                                       DeviceBuffer<scalar>& diag, bool addToDiag = true)
{ deviceAmiAssembleLaplacian(ami, gammaCell, diag, addToDiag); }

// `wsch`: the div scheme's face interpolation weight per interface face. Null (the default) means
// upwind, w = pos0(phi), which is what both assemblies hardcoded for every case regardless of what the
// case's div(phi,U) actually asked for -- see amiMomKernel for the OpenFOAM derivation.
inline void interfaceAssembleMomentum(DeviceCyclic& cyc, const DeviceBuffer<scalar>& nuEffCell,
                                      DeviceBuffer<scalar>& diag,
                                      const DeviceBuffer<scalar>* wsch = nullptr)
{ deviceCyclicAssembleMomentum(cyc, nuEffCell, diag, wsch); }
inline void interfaceAssembleMomentum(DeviceAMI& ami, const DeviceBuffer<scalar>& nuEffCell,
                                      DeviceBuffer<scalar>& diag,
                                      const DeviceBuffer<scalar>* wsch = nullptr)
{ deviceAmiAssembleMomentum(ami, nuEffCell, diag, wsch); }

inline void interfaceOffDiagSum(const DeviceCyclic& cyc, DeviceBuffer<scalar>& sumOff)
{ deviceCyclicOffDiagSum(cyc, sumOff); }
inline void interfaceOffDiagSum(const DeviceAMI& ami, DeviceBuffer<scalar>& sumOff)
{ deviceAmiOffDiagSum(ami, sumOff); }

inline void interfaceScaleImplicit(DeviceCyclic& cyc) { deviceCyclicScaleImplicit(cyc); }
inline void interfaceScaleImplicit(DeviceAMI& ami)    { deviceAmiScaleImplicit(ami); }

// epsilon/omega setValues: zero the interface off-diagonal for fixed wall-cell owners.
inline void interfaceZeroWallIfCoeff(DeviceCyclic& cyc, const DeviceBuffer<label>& isWallCell)
{ deviceCyclicZeroWallIfCoeff(cyc, isWallCell); }
inline void interfaceZeroWallIfCoeff(DeviceAMI& ami, const DeviceBuffer<label>& isWallCell)
{ deviceAmiZeroWallIfCoeff(ami, isWallCell); }

// --- H (reciprocal-diagonal) coupling -------------------------------------------------------------------------
// nbr = the interface neighbour contribution (psi for cyclic, precomputed UkNbr for AMI); H[own] -= ifCoeff*nbr/V.
inline void interfaceAddH(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& nbr,
                          const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& H)
{ deviceCyclicAddH(cyc, nbr, V, H); }
inline void interfaceAddH(const DeviceAMI& ami, const DeviceBuffer<scalar>& nbr,
                          const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& H)
{ deviceAmiAddH(ami, nbr, V, H); }

// --- flux (H . Sf across the interface) + continuity ----------------------------------------------------------
inline void interfaceFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& Hx,
                          const DeviceBuffer<scalar>& Hy, const DeviceBuffer<scalar>& Hz)
{ deviceCyclicFlux(cyc, Hx, Hy, Hz); }
inline void interfaceFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& Hx,
                          const DeviceBuffer<scalar>& Hy, const DeviceBuffer<scalar>& Hz)
{ deviceAmiFlux(ami, Hx, Hy, Hz); }

inline void interfaceAddDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div)
{ deviceCyclicAddDiv(cyc, V, div); }
inline void interfaceAddDiv(const DeviceAMI& ami, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div)
{ deviceAmiAddDiv(ami, V, div); }

// pressure-correction flux: phi -= ifCoeff*(p[nbr]-p[own]) across the interface.
inline void interfaceCorrectFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& p)
{ deviceCyclicCorrectFlux(cyc, p); }
inline void interfaceCorrectFlux(DeviceAMI& ami, const DeviceBuffer<scalar>& p)
{ deviceAmiCorrectFlux(ami, p); }

// --- gradients (gaussGrad interface contribution) -------------------------------------------------------------
inline void interfaceAddGrad(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& psi, const DeviceBuffer<scalar>& V,
                             DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz)
{ deviceCyclicAddGrad(cyc, psi, V, gx, gy, gz); }
inline void interfaceAddGrad(const DeviceAMI& ami, const DeviceBuffer<scalar>& psi, const DeviceBuffer<scalar>& V,
                             DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz)
{ deviceAmiAddGrad(ami, psi, V, gx, gy, gz); }

// divDevReff stress-tensor divergence contribution (raw V*fvc::div, no /V).
inline void interfaceAddTensorDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& sigmaC, int nC,
                                  DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ)
{ deviceCyclicAddTensorDiv(cyc, sigmaC, nC, srcX, srcY, srcZ); }
inline void interfaceAddTensorDiv(const DeviceAMI& ami, const DeviceBuffer<scalar>& sigmaC, int nC,
                                  DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ)
{ deviceAmiAddTensorDiv(ami, sigmaC, nC, srcX, srcY, srcZ); }

// --- rotational / deferred momentum corrections ---------------------------------------------------------------
// deferred convection correction for component comp from the three interface velocity components.
inline void interfaceAddDeferredRot(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& uX,
                                    const DeviceBuffer<scalar>& uY, const DeviceBuffer<scalar>& uZ,
                                    int comp, DeviceBuffer<scalar>& src)
{ deviceCyclicAddDeferredRot(cyc, uX, uY, uZ, comp, src); }
inline void interfaceAddDeferredRot(const DeviceAMI& ami, const DeviceBuffer<scalar>& uX,
                                    const DeviceBuffer<scalar>& uY, const DeviceBuffer<scalar>& uZ,
                                    int comp, DeviceBuffer<scalar>& src)
{ deviceAmiAddDeferredRot(ami, uX, uY, uZ, comp, src); }

// linearUpwind deferred gradient correction at the interface (component comp).
inline void interfaceAddLinUpwindCorr(const DeviceCyclic& cyc, int comp, const DeviceBuffer<scalar>* gUx,
                                      const DeviceBuffer<scalar>* gUy, const DeviceBuffer<scalar>* gUz,
                                      DeviceBuffer<scalar>& corr)
{ deviceCyclicAddLinUpwindCorr(cyc, comp, gUx, gUy, gUz, corr); }
inline void interfaceAddLinUpwindCorr(const DeviceAMI& ami, int comp, const DeviceBuffer<scalar>* gUx,
                                      const DeviceBuffer<scalar>* gUy, const DeviceBuffer<scalar>* gUz,
                                      DeviceBuffer<scalar>& corr)
{ deviceAmiAddLinUpwindCorr(ami, comp, gUx, gUy, gUz, corr); }

// ...and their SCALAR forms, for the turbulence/energy transport.
inline void interfaceAddLinUpwindCorr(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gx,
                                      const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                                      DeviceBuffer<scalar>& corr)
{ deviceCyclicAddLinUpwindCorr(cyc, gx, gy, gz, corr); }
inline void interfaceAddLinUpwindCorr(const DeviceAMI& ami, const DeviceBuffer<scalar>& gx,
                                      const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                                      DeviceBuffer<scalar>& corr)
{ deviceAmiAddLinUpwindCorr(ami, gx, gy, gz, corr); }
inline void interfaceAddLapCorr(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                                const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                                const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corr)
{ deviceCyclicAddLapCorr(cyc, gammaCell, gx, gy, gz, corr); }
inline void interfaceAddLapCorr(const DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell,
                                const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                                const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corr)
{ deviceAmiAddLapCorr(ami, gammaCell, gx, gy, gz, corr); }

// velocity non-orthogonal Laplacian correction (component comp, rotated neighbour reconstruction).
inline void interfaceAddLapCorr(const DeviceCyclic& cyc, int comp, const DeviceBuffer<scalar>& gammaCell,
                                const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy,
                                const DeviceBuffer<scalar>* gUz, DeviceBuffer<scalar>& corr)
{ deviceCyclicAddLapCorr(cyc, comp, gammaCell, gUx, gUy, gUz, corr); }
inline void interfaceAddLapCorr(const DeviceAMI& ami, int comp, const DeviceBuffer<scalar>& gammaCell,
                                const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy,
                                const DeviceBuffer<scalar>* gUz, DeviceBuffer<scalar>& corr)
{ deviceAmiAddLapCorr(ami, comp, gammaCell, gUx, gUy, gUz, corr); }

// pressure non-orthogonal correction (scalar p): -ffc into bp[own]; ffcOut holds ffc for post-solve flux fix.
inline void interfaceLapCorrP(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                              const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                              const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& bp, DeviceBuffer<scalar>& ffcOut)
{ deviceCyclicLapCorrP(cyc, gammaCell, gx, gy, gz, bp, ffcOut); }
inline void interfaceLapCorrP(const DeviceAMI& ami, const DeviceBuffer<scalar>& gammaCell,
                              const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                              const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& bp, DeviceBuffer<scalar>& ffcOut)
{ deviceAmiLapCorrP(ami, gammaCell, gx, gy, gz, bp, ffcOut); }

} // namespace brae
