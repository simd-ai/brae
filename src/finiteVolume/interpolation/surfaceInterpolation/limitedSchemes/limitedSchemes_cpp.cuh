#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's TVD-limited and blended convection schemes.
//
// provenance:
//   openfoam:
//     limitedSurfaceInterpolationScheme.C::weights   weights = limiter*CDweights + (1-limiter)*pos0(phi)
//     limitedLinear.H                                limiter = clamp(twoByk*r, 0, 1),  twoByk = 2/max(k,SMALL)
//     NVDTVD.H::r        (scalar)                    r = 2*(gradcf/gradf) - 1
//     NVDVTVDV.H::r      (the `V` variants)          gradf = gradfV & gradfV,  gradcf = gradfV & (d & gradc)
//     LUST.H                                         weights = 0.75*linear + 0.25*upwind
//                                                    correction = 0.25*linearUpwind::correction
//     linearUpwindV.C                                the vector-LIMITED linearUpwind correction
//   brae:
//     reference: src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cu
//     cuda:      src/cuda/device_deferred_correction.cu (deviceDivLimitedCoeffs, deviceLinearUpwindVCorr,
//                                                        deviceLinearCorr)
//     tests:     tests/test_limitedschemes_cpp.cu
//
// EVERY ONE OF THESE SCHEMES IS A WEIGHT CHANGE, A DEFERRED CORRECTION, OR BOTH, and which of the three
// it is decides how it is ported:
//
//   upwind          weights = pos0(phi)                       no correction
//   linearUpwind    weights = pos0(phi)  (derives from upwind) correction from grad(vf)
//   limitedLinear   weights = limiter*CD + (1-limiter)*pos0    no correction
//   LUST            weights = 0.75*CD + 0.25*pos0             correction = 0.25*linearUpwind's
//
// Getting that classification wrong is silent: a scheme ported as "upwind plus a correction" when it is
// really a weight change still converges, and to a plausible answer.
//
// A THIRD BEHAVIOUR, and the easiest of the three to miss: `limitedLinear` APPLIED TO A VECTOR is not
// per-component and is not the V form either. LimitedScheme.H's family macro instantiates it as
// NVDTVD + limitFuncs::magSqr, so the limiter is built from the SCALAR field magSqr(U) and its gradient,
// and only `limitedLinearV` uses NVDVTVDV. So on U:
//     limitedLinear   -> limitedLinearWeights (below) with vf = magSqr(U), gradVf = grad(magSqr(U))
//     limitedLinearV  -> limitedLinearVWeights
// Three plausible readings, one right; the other two converge to plausible wrong answers.
//
// `d` IS THE OWNER-TO-NEIGHBOUR VECTOR, C[nei] - C[own], and the r ratio branches on faceFlux to pick
// which cell's gradient it uses -- so r is NOT symmetric in the two cells and the sign of phi matters.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include <cmath>
#include <vector>

namespace brae {
namespace cpu {
namespace limitedSchemes {

// THE LIMITER MATH ITSELF, exposed rather than kept private to the .cu, because it is needed in TWO
// places that must not drift: the internal faces below, and the COUPLED (cyclic / cyclicAMI) faces,
// which OpenFOAM limits with the same functions in LimitedScheme::calcLimiter's coupled branch. A
// second transcription for the interface would be a second chance to get NVDTVD's 1000x guard, its
// >= vs >, or OpenFOAM's sign() convention wrong -- so there is one.
namespace detail {

inline scalar sign0(scalar x) { return (x >= 0) ? 1.0 : -1.0; }   // OpenFOAM sign(): >= 0 -> +1

inline scalar clamp01(scalar x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

// NVDTVD::r -- the scalar TVD ratio. `gradf` is the face difference, `gradcf` the projection of the
// UPWIND cell's gradient onto d. The 1000x guard is OpenFOAM's, not a regularisation of our own.
inline scalar rScalar(scalar faceFlux, scalar phiP, scalar phiN,
                      const vector& gradcP, const vector& gradcN, const vector& d)
{
    const scalar gradf = phiN - phiP;
    const vector& gc = (faceFlux > 0) ? gradcP : gradcN;
    const scalar gradcf = d.x*gc.x + d.y*gc.y + d.z*gc.z;
    if (std::fabs(gradcf) >= 1000.0*std::fabs(gradf))
        return 2.0*1000.0*sign0(gradcf)*sign0(gradf) - 1.0;
    return 2.0*(gradcf/gradf) - 1.0;
}

// NVDVTVDV::r -- the V form. gradf is the SQUARED magnitude of the vector difference and gradcf is that
// difference dotted with (d & gradc), so one limiter serves all three components. That coupling is the
// whole difference between `limitedLinear` and `limitedLinearV`.
//
// gradcP/gradcN are in OPENFOAM's tensor packing, gradc_ij = d(U_j)/d(x_i), so the projection (d & gradc)
// is a COLUMN dot. brae's device packs its velocity gradient the other way round (row = component), and
// the two are transposes of each other -- a caller holding the device packing must transpose, not just
// pass it through. See cyclicAMI_cpp::limitedLinearVWeights, which does exactly that and says so.
inline scalar rVector(scalar faceFlux, const vector& phiP, const vector& phiN,
                      const tensor& gradcP, const tensor& gradcN, const vector& d)
{
    const vector gradfV { phiN.x - phiP.x, phiN.y - phiP.y, phiN.z - phiP.z };
    const scalar gradf = gradfV.x*gradfV.x + gradfV.y*gradfV.y + gradfV.z*gradfV.z;
    const tensor& gc = (faceFlux > 0) ? gradcP : gradcN;
    // (d & gradc)_j = d_i * gradc_ij
    const vector dg { d.x*gc.xx + d.y*gc.yx + d.z*gc.zx,
                      d.x*gc.xy + d.y*gc.yy + d.z*gc.zy,
                      d.x*gc.xz + d.y*gc.yz + d.z*gc.zz };
    const scalar gradcf = gradfV.x*dg.x + gradfV.y*dg.y + gradfV.z*dg.z;
    if (std::fabs(gradcf) >= 1000.0*std::fabs(gradf))
        return 2.0*1000.0*sign0(gradcf)*sign0(gradf) - 1.0;
    return 2.0*(gradcf/gradf) - 1.0;
}

// limitedLinear.H's limiter, and the blend limitedSurfaceInterpolationScheme::weights applies to it.
inline scalar limitedLinearLimiter(scalar r, scalar twoByk) { return clamp01(twoByk * r); }

inline scalar blend(scalar limiter, scalar cdWeight, scalar faceFlux)
{
    return limiter*cdWeight + (1.0 - limiter)*((faceFlux >= 0.0) ? 1.0 : 0.0);   // pos0
}

} // namespace detail


// THE SAME SCHEMES AT A COUPLED PATCH FACE (cyclic, cyclicAMI).
//
// OpenFOAM limits a coupled patch exactly as it limits an internal face -- LimitedScheme::calcLimiter
// has a `coupled()` branch that calls the SAME Limiter::limiter with patch-side arguments, and only an
// UNCOUPLED patch gets the constant limiter of 1.0. The four substitutions are:
//
//     CDweights[face]  ->  the patch's own interpolation weight  (fvPatch::weights)
//     C[nei] - C[own]  ->  fvPatch::delta(), which for a transformed patch is
//                          dOwn - transform(forwardT, interpolated dNbr) -- NOT a cell-centre difference
//     lPhi[own/nei]    ->  patchInternalField / patchNeighbourField
//     gradc[own/nei]   ->  the gradient's patchInternalField / patchNeighbourField
//
// and patchNeighbourField is where the transform lives: a scalar is never transformed, a vector is
// rotated, a tensor is rotated on BOTH indices. The caller supplies both sides already prepared, so
// these functions are pure per-face arithmetic and carry no interface addressing.
//
// WHY THIS MATTERS ENOUGH TO PORT. Without it brae assembles every coupled face upwind whatever the case
// asked for, because w = pos0(phi) is what upwind's max/min split IS. pipeCyclic asks for
// `bounded Gauss limitedLinearV 1`, and against OpenFOAM's own boundaryCoeffs the interface off-diagonal
// came out at L2 6.86e-01 -- brae's coefficient never changing sign where OpenFOAM's does.
std::vector<scalar> limitedLinearWeightsCoupled(
    const std::vector<scalar>& phi,   // the interface face flux
    const std::vector<scalar>& cd,    // the patch's interpolation weights
    const std::vector<vector>& d,     // fvPatch::delta()
    const std::vector<scalar>& vfP,   // patchInternalField
    const std::vector<scalar>& vfN,   // patchNeighbourField
    const std::vector<vector>& gP,    // grad's patchInternalField
    const std::vector<vector>& gN,    // grad's patchNeighbourField (rotated when the patch transforms)
    scalar                     k);

// The V form. gP/gN use OPENFOAM's packing, gradc_ij = d(U_j)/d(x_i) -- see detail::rVector.
std::vector<scalar> limitedLinearVWeightsCoupled(
    const std::vector<scalar>& phi,
    const std::vector<scalar>& cd,
    const std::vector<vector>& d,
    const std::vector<vector>& vfP,
    const std::vector<vector>& vfN,
    const std::vector<tensor>& gP,
    const std::vector<tensor>& gN,
    scalar                     k);


// pos0(phi): the upwind weights. 1 where the flux leaves the owner, 0 where it enters.
std::vector<scalar> upwindWeights(const std::vector<scalar>& phi);

// LUST: 0.75*linear + 0.25*upwind. LUST also carries 0.25 of linearUpwind's deferred correction, which is
// the caller's job -- LUST.H overrides BOTH weights() and correction() and a port that took only one of
// them would be a different scheme.
std::vector<scalar> lustWeights(const std::vector<scalar>& phi, const FvGeometry& g);

// limitedLinear, scalar form: weights = limiter*CD + (1-limiter)*pos0(phi).
std::vector<scalar> limitedLinearWeights(
    const std::vector<scalar>&        phi,
    const GeometricField<scalar>&     vf,
    const std::vector<vector>&        gradVf,
    scalar                            k,        // the scheme coefficient; `limitedLinear 1` -> k = 1
    const PrimitiveMesh&              m,
    const FvGeometry&                 g);

// limitedLinearV: the same limiter, but r is formed on the VECTOR difference (NVDVTVDV), so all three
// components share one limiter per face instead of being limited independently.
std::vector<scalar> limitedLinearVWeights(
    const std::vector<scalar>&        phi,
    const GeometricField<vector>&     vf,
    const std::vector<tensor>&        gradVf,
    scalar                            k,
    const PrimitiveMesh&              m,
    const FvGeometry&                 g);

// linearUpwindV's DEFERRED CORRECTION, as an extensive per-cell contribution (linearUpwindV.C).
//
// linearUpwindV derives from `upwind` exactly as linearUpwind does -- same weights, correction only --
// but the correction is a DIFFERENT function, not a scaled one, so it cannot be folded into a factor:
//
//     phi_f > 0:  corr = (Cf - C[own]) & grad(vf)[own],   maxCorr = (1-w)*(vf[nei] - vf[own])
//     phi_f < 0:  corr = (Cf - C[nei]) & grad(vf)[nei],   maxCorr =    w *(vf[own] - vf[nei])
//     s = magSqr(corr);  mx = corr & maxCorr
//     if (s > 0) { if (mx < 0) corr = 0;  else if (s > mx) corr *= mx/(s + VSMALL); }
//
// The limiter is what distinguishes it: the correction may not overshoot the owner-to-neighbour jump
// PROJECTED ON ITS OWN DIRECTION, and it is zeroed outright when it points against that jump. That test
// couples the three components -- magSqr and the dot product are over the vector -- so this cannot be
// applied per component, which is the same trap `limitedLinear` vs `limitedLinearV` sets.
//
// Returned as faceSum(phi_f * corr_f) per cell, the form the caller SUBTRACTS (fvMatrix::operator+= is
// `source -= V*su`), matching fvm::linearUpwindCorrection.
// The FACE corrections, before they are summed into cells. Exposed because the limiter has an exact
// postcondition that is only checkable per face -- after limiting, either corr == 0 or
// 0 <= magSqr(corr) <= (corr & maxCorr) -- and the per-cell sum below is computed FROM this, so a test
// of one is a test of the other.
std::vector<vector> linearUpwindVFaceCorrection(
    const std::vector<scalar>&    phi,
    const GeometricField<vector>& vf,
    const std::vector<tensor>&    gradVf,
    const PrimitiveMesh&          m,
    const FvGeometry&             g);

std::vector<vector> linearUpwindVCorrection(
    const std::vector<scalar>&    phi,
    const GeometricField<vector>& vf,
    const std::vector<tensor>&    gradVf,
    const PrimitiveMesh&          m,
    const FvGeometry&             g);

} // namespace limitedSchemes
} // namespace cpu
} // namespace brae
