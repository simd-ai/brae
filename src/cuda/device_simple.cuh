#pragma once
// cf GPU offload (G7): the SIMPLE p-U coupling glue on device. Given the momentum matrix (G6) these build
// H(), rAU=1/A, the HbyA flux phiHbyA, the pressure flux pEqn.flux(), and the velocity corrector, every
// quantity the SIMPLE step needs between the momentum and pressure solves, validated against the CPU.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"

namespace brae {

// H() per component: H = (source + lduMatrix::H(psi) + boundary-diag term + boundarySource)/V.
// bdDiagK = (cmptAv(ic)-ic_k) per boundary face; bdSrcK = boundaryCoeffs_k per boundary face (host-flattened).
void deviceMatrixH(const DeviceLduView& A, const DeviceMesh& dm, const DeviceBuffer<scalar>& psiK,
                   const DeviceBuffer<scalar>& sourceK, const DeviceBuffer<scalar>& bdDiagK,
                   const DeviceBuffer<scalar>& bdSrcK, DeviceBuffer<scalar>& Hk);

// rAU = V/diagC  (A = diagC/V, rAU = 1/A). diagC already folded (= diag + internalCoeffs cmptAv).
void deviceReciprocalV(const DeviceMesh& dm, const DeviceBuffer<scalar>& diagC, DeviceBuffer<scalar>& rAU);
// SIMPLEC consistent rAtU = V/rowSum, rowSum = A*ones. This is OF's 1/(1/rAU - UEqn.H1()) exactly:
// H1 = (-sum(offdiag) + sum(coupled boundaryCoeffs))/V and A = D/V, so V*(A - H1) = D + sum(offdiag)
// - sum(coupled bCoeffs) = rowSum on a mesh with no coupled patches. diagA is unused (kept for ABI).
// NOTE: there is NO max(rowSum, 0.1*diagA) clamp -- an earlier version of this comment claimed one and
// OF has none; the code never had it either (diagA is (void)-cast).
void deviceSimplecRAtU(const DeviceMesh& dm, const DeviceBuffer<scalar>& rowSum, const DeviceBuffer<scalar>& diagA,
                       DeviceBuffer<scalar>& rAtU);
// linearUpwind deferred correction for div(phi,U_i): corrSource[c] = Sum(+/-) phi_f*(grad(U_i)_upwind . (Cf-C_up)).
void deviceLinearUpwindCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& gx,
                            const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corrSource);
// linearUpwindV (OF vector-limited linearUpwind): all 3 components at once (the limiter couples them). gUx/gUy/gUz are
// arrays [3] of the per-component gradients (gUx[i]=d(U_i)/dx). Produces the 3 component correction sources.
void deviceLinearUpwindVCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt,
                             const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy, const DeviceBuffer<scalar>* gUz,
                             const DeviceBuffer<scalar>& U0, const DeviceBuffer<scalar>& U1, const DeviceBuffer<scalar>& U2,
                             DeviceBuffer<scalar>& corrX, DeviceBuffer<scalar>& corrY, DeviceBuffer<scalar>& corrZ);
// LUST linear part: div(phi*(linear-upwind)_face). OF LUST = 0.75*deviceLinearCorr + 0.25*deviceLinearUpwindCorr.
void deviceLinearCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& field,
                      DeviceBuffer<scalar>& corrSource);

// phiHbyA internal = flux of the HbyA vector (SoA components) = (interp(HbyA) & Sf).
void deviceVectorFlux(const DeviceMesh& dm, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                      const DeviceBuffer<scalar>& Uz, DeviceBuffer<scalar>& phiInt);

// pEqn.flux() internal = upper*p[nei] - lower*p[own].
void deviceMatrixFluxInternal(const DeviceLduView& A, const DeviceBuffer<scalar>& p, DeviceBuffer<scalar>& fluxInt);

// velocity corrector component: U = HbyA - rAU*gradP.
void deviceCorrector(const DeviceBuffer<scalar>& HbyA, const DeviceBuffer<scalar>& rAU,
                     const DeviceBuffer<scalar>& gradP, DeviceBuffer<scalar>& U);

// boundary flux: phiHbyA.boundary = (HbyA.boundary & Sf) per boundary face (SoA boundary values + Sf gather).
void deviceBoundaryFlux(const DeviceMesh& dm, const DeviceBuffer<scalar>& uxb, const DeviceBuffer<scalar>& uyb,
                        const DeviceBuffer<scalar>& uzb, DeviceBuffer<scalar>& phiB);

// adjustPhi (no-pressure-reference cases): scale the ADJUSTABLE outflow so the net boundary flux = 0
// (global continuity). adjustable[i]=1 if the U patch does NOT fix the flux (zeroGradient/inletOutlet/calculated),
// 0 if it does (fixedValue/noSlip). massCorr = (massIn - fixedMassOut)/adjustableMassOut. Returns massCorr; a no-op
// (massCorr=1) for a closed domain (all fixed, no adjustable outflow). Operates on the boundary flux in place.
// OF adjustPhi. phiInt is the INTERNAL flux and is needed for OF's normaliser
// totalFlux = VSMALL + sum(mag(phi)), which is taken over the whole surface field and not just the
// boundary slice being scaled. Null keeps the old behaviour of normalising on the boundary alone, which
// makes the relative guard weaker than OpenFOAM's -- pass it wherever the internal flux is in hand.
scalar deviceAdjustPhi(const DeviceBuffer<label>& adjustable,
                       DeviceBuffer<scalar>& phiB,
                       const DeviceBuffer<scalar>* phiInt = nullptr);

// fvMatrix::setReference (no-pressure-reference cases): pin the singular pressure system at refCell.
// OF: source[ref] += diag[ref]*refValue; diag[ref] += diag[ref]. Call BEFORE the AMG coarsening + normFactor.
void deviceSetReference(DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& b, label refCell, scalar refValue);

// fvMatrix::relax, relaxedDiag[c] = max(|diag + sum|ic||, sumOff)/alpha - sum(ic) ; delta = relaxedDiag-diag.
// iCbnd = the (isotropic) boundary internalCoeffs scalar; the per-component source gains delta*psi (caller).
// cycSumOff (optional): per-cell sum of |cyclic off-diagonal| (deviceCyclicOffDiagSum) added to sumOff so the
// periodic interface counts toward diagonal dominance (exact x-invariance on periodic meshes). nullptr -> ignored.
// iCmaxMag (optional): per-boundary-face cmptMax(cmptMag(iC0,iC1,iC2)), the max-over-components boundary diagonal
// magnitude (OF fvMatrix::relax adds this to the diagonal). When given it replaces |iCbnd| in the dominance/relax
// (and hence rAU) sum; needed where the boundary diagonal differs per component (slip/symmetry). nullptr -> |iCbnd|.
// iCmin: OF's relax REMOVES cmptMin(iC) after the relax, which is NOT the same quantity it added -- the add uses
// cmptMax(cmptMag(iC)), the remove uses the signed cmptMin across components. They agree only when the three
// components are equal, so this matters exactly where iC is per-component (slip/symmetry). nullptr -> iCbnd.
void deviceRelaxDiag(const DeviceLduView& A, const DeviceMesh& dm, const DeviceBuffer<scalar>& iCbnd, scalar alpha,
                     DeviceBuffer<scalar>& relaxedDiag, DeviceBuffer<scalar>& delta, const scalar* cycSumOff = nullptr,
                     const scalar* iCmaxMag = nullptr, const scalar* iCmin = nullptr);

// per-element out[i] = max(|a[i]|, |b[i]|, |c[i]|) (cmptMax(cmptMag) across the 3 momentum components' boundary iC).
void deviceCmptMaxMag3(const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& c,
                       DeviceBuffer<scalar>& out);

// per-element out[i] = min(a[i], b[i], c[i]) -- SIGNED, matching OF cmptMin (not cmptMin of magnitudes).
void deviceCmptMin3(const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& c,
                    DeviceBuffer<scalar>& out);

} // namespace brae
