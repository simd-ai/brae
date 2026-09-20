#pragma once
// interFoam's alpha fluxes on the device -- phic, phir, alphaPhiUn and rhoPhi.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/VoF/alphaEqn.H:59-89 (phic), :162 (phir),
//              :164-176 (alphaPhiUn), :248 (rhoPhi)
//   host:      src/applications/solvers/interFoam/alpha_eqn_cpp.cu -- the ORACLE, gated against
//              OpenFOAM's own expressions in tests/test_alpha_eqn_cpp.cu.
//   tests:     tests/test_device_alpha_flux.cu
//
// THE NESTED FLUX IS COMPOSED ON THE HOST, ONE LAUNCH PER TERM, and that is deliberate:
//
//     alphaPhiUn = fvc::flux(phi, alpha1, alphaScheme)
//                + fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme), alpha1, alpharScheme)
//
// is five face-sized operations with TWO MINUS SIGNS in the middle of them, and the upwind direction
// of each interpolation is set by the flux passed in -- so the negations are not cosmetic, they change
// which cell each face reads. Fusing it into one kernel would hide exactly that, and the arithmetic
// intensity is far too low for the fusion to pay: every one of these is memory-bound on a face field.
// The host composition stays readable and each piece is separately gated.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"

namespace brae {

// fvc::flux(psi, vf, scheme) == psi*interpolate(vf), with the interpolation weights `w` computed from
// THAT psi (deviceLimitedFaceWeights for vanLeer, the mesh's own for linear, pos0 for upwind). The
// weights are an argument rather than a scheme enum so the caller cannot pass one flux to the weights
// and another to the multiply -- which is a different operator, and the one mistake this call invites.
// ...and on the faces of a periodic pair, where the scheme's weight applies as on an internal face.
// `scheme`: 0 linear, 1 upwind, 2 vanLeer. interfaceCompression across a coupled patch is not ported --
// the host arm refuses it by name (alpha_eqn_cpp.cu:286-289) and so must any caller of this.
//
// THE GRADIENT HANDED IN MUST ALREADY CARRY THE PAIR (deviceCyclicAddGrad after deviceGaussGrad). The
// device mesh keeps a cyclic patch out of its boundary gather, so a plain deviceGaussGrad is the
// gradient of a mesh with a WALL there, and the vanLeer limiter reads it in exactly the cells next to
// the pair. MEASURED with the plain gradient: linear and upwind still exact, vanLeer 1.6e-02 of a
// 3.9e-02 flux. There is nothing here that can check it, which is why it is stated.
void deviceAlphaCyclicFlux(
    const DeviceCyclic&         cyc,
    int                         scheme,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>&       out);

void deviceAlphaFaceFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& psiInt,
    const DeviceBuffer<scalar>& w,
    const DeviceBuffer<scalar>& field,
    DeviceBuffer<scalar>&       out);

// out = -in, per face. Exists so the two minus signs in alphaPhiUn are visible as two calls rather
// than folded into a sign flag nobody reads.
void deviceNegateFaces(int n, const DeviceBuffer<scalar>& in, DeviceBuffer<scalar>& out);

// phic = cAlpha*|phi/magSf|, alphaEqn.H:59 -- and ZERO on every non-coupled boundary face (:79-89).
// Interface compression is anti-diffusion; at an open boundary it sharpens an interface the boundary
// does not have, so OpenFOAM switches it off there and so does this.
void deviceCompressionFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    int                         nBoundaryFaces,
    const DeviceBuffer<scalar>& phiInt,
    scalar                      cAlpha,
    DeviceBuffer<scalar>&       phicInt,
    DeviceBuffer<scalar>&       phicBnd);

// rhoPhi = alphaPhi10*(rho1 - rho2) + phiForRho2*rho2, alphaEqn.H:248. The subtraction ORDER is what
// makes this reduce to phi*rho1 in pure phase 1 and phi*rho2 in pure phase 2; swapped it keeps both
// the dimensions and the magnitude and inverts the mixture.
void deviceMassFlux(
    int                         n,
    const DeviceBuffer<scalar>& alphaPhi,
    const DeviceBuffer<scalar>& phiForRho2,
    scalar                      rho1,
    scalar                      rho2,
    DeviceBuffer<scalar>&       rhoPhi);

// phir = phic*mixture.nHatf(), alphaEqn.H:162. nHatf is already a FLUX (nHatfv & Sf), not a unit
// vector, so this is a product of two face fields and needs no area weighting of its own.
void deviceMultiplyFaces(int n, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b,
                         DeviceBuffer<scalar>& out);

// out = a - b, per face. phiCorr = phiPsi - phiBD.
void deviceSubtractFaces(int n, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b,
                         DeviceBuffer<scalar>& out);

} // namespace brae
