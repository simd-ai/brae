#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's cyclicAMI coupling as the finite-volume layer uses it.
//
// provenance:
//   openfoam:
//     files:   src/finiteVolume/fields/fvPatchFields/constraint/cyclicAMI/cyclicAMIFvPatchField.C
//              src/meshTools/AMIInterpolation/AMIInterpolation/AMIInterpolation.C
//     symbols: cyclicAMIFvPatchField::updateInterfaceMatrix / patchNeighbourField,
//              AMIInterpolation::interpolateToSource
//   brae:
//     geometry:  src/OpenFOAM/meshes/interface/ami_interface.cuh  (AMIInterface: weights, addressing,
//                deltaCoeffs, corrVec, forwardT -- already validated by ami_weights/ami_geometry)
//     reference: src/finiteVolume/fvMatrices/cyclicAMI/cyclicAMI_cpp.cu
//     cuda:      src/cuda/device_ami.cu
//     tests:     tests/ami_cpp_vs_device.cu
//
// WHY THIS EXISTS. brae's device AMI is one fused path, and every attempt to explain pipeCyclic's
// disagreement with OpenFOAM has run into the same wall: a single number for the whole interface cannot
// say WHICH stage is wrong. Every other defect in this port fell out quickly once a _cpp reference
// existed to compare against stage by stage -- the pitzDaily inlet diffusivity, the Spalding wall seed,
// turbineSiting's profile origin, the LM lambda loop. The AMI had no such reference, so its residual
// (97% of pipeCyclic's momentum residual sits on interface cells, with the interior exactly zero) has
// stayed unexplained through four passes of reading the device code.
//
// THE STAGES ARE THE DEVICE'S STAGES, deliberately: each function below is the host twin of one device
// entry point, takes the same inputs and returns the same outputs, so a disagreement names a kernel.
//
//   interpolate      <-> deviceAmiInterpolate      out[i]   = sum_k w[k]*psi[nbr[k]]            (no transform)
//   interpolateVec   <-> deviceAmiInterpolateVec   out[i]   = sum_k w[k]*(forwardT & U[nbr[k]])
//   faceValue        <-> deviceAmiFaceValue        out[i]   = w_i*psi[own] + (1-w_i)*interp[i]
//   assembleMomentum <-> deviceAmiAssembleMomentum ifCoeff  = -lap + phi*(1-w);  diag += lap + phi*w
//                                                  w = the DIV SCHEME's face weight; upwind is the
//                                                  special case w = pos0(phi)
//                                                  lap      = (w*nuEff[own] + (1-w)*interp(nuEff))*dc*magSf
//   assembleLaplacian<-> deviceAmiAssembleLaplacian ifCoeff = +c;                  diag -= c
//                                                  (the OPPOSITE sign to momentum: the pressure equation
//                                                   carries +laplacian(rAUf,p), momentum -laplacian(nuEff,U))
//   amul             <-> deviceAmiAmul             Apsi[own] += ifCoeff[i]*interp(psi)[i]
//   addH             <-> deviceAmiAddH             H[own]    -= ifCoeff[i]*UN[i]/V[own]
//   flux             <-> deviceAmiFlux             phi[i]    = faceValue(HbyA) . Sf[i]
//   correctFlux      <-> deviceAmiCorrectFlux      phi[i]   -= ifCoeff[i]*(interp(p)[i] - p[own])
//   addDiv           <-> deviceAmiAddDiv           div[own] += phi[i]/V[own]
//   addGrad          <-> deviceAmiAddGradRot       grad[own] += Sf[i]*(w*U[own] + (1-w)*UN[i])/V[own]
//   lapCorr          <-> deviceAmiAddLapCorr       src[own] -= gammaf*magSf*(corrVec . gradFace)
//                                                  gradFace = w*grad(U_c)[own] + (1-w)*sum_k w_k (R G_k R^T)[c]
//
// THE TRANSFORM IS APPLIED BEFORE THE WEIGHTED SUM, not after -- forwardT is constant per interface so
// the two agree, but the device does it that way and a reference that reorders them stops being a
// reference for the thing it is checking.
//
// ONE DIRECTION PER ENTRY. An interface entry couples own <- nbr and nothing else; the reverse coupling
// is a separate entry on the paired patch. That is why amul ADDS to Apsi[own] only, and why the AMG
// agglomeration gives each appended edge upper = ifCoeff with lower = zero.
#include "cf_types.cuh"
#include "ami_interface.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"

#include <vector>

namespace brae {
namespace cpu {
namespace cyclicAMI {

// Per-interface working state: the assembled off-diagonal and face flux, mirroring DeviceAMI's
// ifCoeff/phi so a comparison is field-for-field.
struct State
{
    std::vector<scalar> ifCoeff;   // per src face
    std::vector<scalar> phi;       // per src face
};

std::vector<scalar> interpolate(const AMIInterface& a, const std::vector<scalar>& psi);

std::vector<vector> interpolateVec(const AMIInterface& a, const std::vector<vector>& U);

std::vector<scalar> faceValue(const AMIInterface& a, const std::vector<scalar>& cell);

// patchNeighbourField for a cell TENSOR field. A rotational interface transforms a tensor on BOTH
// indices, R G R^T -- the same rule lapCorr uses below, and the reason it is spelled out there.
//
// The formula is INDIFFERENT to which of the two gradient packings the caller holds: transposing G
// transposes the result, R G^T R^T = (R G R^T)^T, so a device-packed gradient in gives a device-packed
// gradient out. That is worth stating because almost nothing else about these two packings is safe.
std::vector<tensor> interpolateTensor(const AMIInterface& a, const std::vector<tensor>& G);

// The DIV SCHEME's face interpolation weight at the interface faces, for a case whose div(phi,U) is
// `Gauss limitedLinearV k`. Feeds assembleMomentum's `wsch`; without it the interface is assembled
// upwind whatever the case asked for.
//
// gradU is in the DEVICE's packing -- (row, col) = (velocity component, derivative direction) -- which is
// the transpose of what OpenFOAM's fvc::grad returns and the transpose of what NVDVTVDV::r expects. The
// transpose happens once, inside, and is the only place in this path it happens.
std::vector<scalar> limitedLinearVWeights(const AMIInterface&        a,
                                          const std::vector<scalar>& phi,      // interface flux, per src face
                                          const std::vector<vector>& U,        // cell velocity
                                          const std::vector<tensor>& gradU,    // cell grad(U), device packing
                                          scalar                     k);

// ...and for `Gauss limitedLinear k` on a SCALAR transport (k, epsilon, omega, nuTilda, gammaInt).
// The value is not transformed across a rotational interface (a scalar has no orientation); its GRADIENT
// is, as a vector. Getting that pair backwards is invisible on a translational interface.
std::vector<scalar> limitedLinearWeights(const AMIInterface&        a,
                                         const std::vector<scalar>& phi,
                                         const std::vector<scalar>& f,
                                         const std::vector<vector>& gradF,
                                         scalar                     k);

// OF's momentum interface: the laplacian face coefficient plus the upwind convective split.
void assembleMomentum(const AMIInterface&        a,
                      const std::vector<scalar>& nuEffCell,
                      const std::vector<scalar>& phi,        // the interface flux, per src face
                      State&                     st,
                      std::vector<scalar>&       diag,
                      // The div scheme's face interpolation weight. Null = upwind (pos0(phi)).
                      const std::vector<scalar>* wsch = nullptr);

// The pressure (laplacian) interface: no convection.
void assembleLaplacian(const AMIInterface&        a,
                       const std::vector<scalar>& gammaCell,
                       State&                     st,
                       std::vector<scalar>&       diag,
                       bool                       addToDiag = true);

// The matrix action: OF cyclicAMIFvPatchField::updateInterfaceMatrix.
void amul(const AMIInterface&        a,
          const State&               st,
          const std::vector<scalar>& psi,
          std::vector<scalar>&       Apsi);

// UEqn.H(): the interface's explicit contribution, per component of an already-interpolated neighbour.
void addH(const AMIInterface&        a,
          const State&               st,
          const std::vector<scalar>& UN,      // interpolateVec's component for this comp
          const std::vector<scalar>& V,
          std::vector<scalar>&       H);

// phiHbyA on the interface faces.
void flux(const AMIInterface&        a,
          const std::vector<vector>& HbyA,
          State&                     st);

// phi -= pEqn.flux() across the interface.
void correctFlux(const AMIInterface&        a,
                 const State&               st,
                 const std::vector<scalar>& p,
                 State&                     phiOut);

// gaussGrad's interface contribution for ONE component: the face value scattered onto the owner cell.
// UN is the ALREADY-INTERPOLATED neighbour component (rotated when the interface is rotational), because
// the device computes the whole rotated vector once before its per-component loop rather than per
// component -- doing it per component would read freshly-updated earlier components.
void addGrad(const AMIInterface&        a,
             const std::vector<scalar>& Uown,
             const std::vector<scalar>& UN,
             const std::vector<scalar>& V,
             std::vector<vector>&       grad);

// The laplacian's DEFERRED non-orthogonal correction across the interface, for component `comp`.
//
// The neighbour's velocity GRADIENT is a tensor, so a rotational interface transforms it as R G R^T --
// not R G, and not G. Getting that wrong is invisible on a translational interface and wrong by the
// sector angle on a rotational one.
void lapCorr(const AMIInterface&        a,
             int                        comp,
             const std::vector<scalar>& gammaCell,
             const std::vector<tensor>& gradU,     // gradU[c] row i = d(U_i)/d(x_.) as the device packs it
             std::vector<scalar>&       corr);

// continuity: the interface's share of div(phi).
void addDiv(const AMIInterface&        a,
            const State&               st,
            const std::vector<scalar>& V,
            std::vector<scalar>&       div);

} // namespace cyclicAMI
} // namespace cpu
} // namespace brae
