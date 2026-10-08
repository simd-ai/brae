#pragma once
// The two-phase mixture on the device -- rho, mu and nu from alpha, every corrector.
//
// provenance:
//   openfoam:  src/transportModels/twoPhaseMixture/twoPhaseMixture/twoPhaseMixture.C
//              src/transportModels/incompressible/incompressibleTwoPhaseMixture/
//                incompressibleTwoPhaseMixture.C:55 (nu), :137 (mu)
//              applications/solvers/multiphase/VoF/alphaEqnSubCycle.H:36 (rho)
//   host:      src/transportModels/twoPhaseMixture/two_phase_mixture_cpp.cuh -- the ORACLE for this
//              file, and it is gated against OpenFOAM's own expressions in
//              tests/test_two_phase_mixture.cu.
//   tests:     tests/test_device_two_phase_mixture.cu (device vs host, bit for bit)
//
// WHY THIS IS THE FIRST THING ON THE DEVICE. mixture.correct() runs once per alpha corrector, once per
// sub-cycle, and again between the sub-cycle and UEqn -- on capillaryRise's settings that is four times
// a step before the momentum equation is even assembled. It is also pure per-cell arithmetic with no
// addressing at all, so it is the piece where a device port can be held to BIT-FOR-BIT agreement with
// the host rather than to a tolerance, which is what makes it a good first one: any difference is a
// defect, not a discretisation.
//
// THE RAW/CLAMPED SPLIT SURVIVES THE PORT, and it is the one thing to get wrong here:
//     rho = alpha1*rho1 + alpha2*rho2                     RAW alpha
//     mu  = a*rho1*nu1 + (1-a)*rho2*nu2,  a = clamp(alpha1)   CLAMPED
//     nu  = mu/(a*rho1 + (1-a)*rho2)                          CLAMPED
// MULES holds alpha in [0,1] to round-off but not exactly, so the two agree on every smooth field and
// differ precisely on the overshoot MULES is allowed to leave. A fused kernel makes it easy to clamp
// once and use the result for all three; that would be wrong, and the gate sweeps alpha outside [0,1]
// on purpose because nothing else can see it.
#include "cf_types.cuh"

namespace brae {

struct DevicePhaseProperties
{
    scalar rho1 = 0, nu1 = 0;
    scalar rho2 = 0, nu2 = 0;
};

// mixture.correct(): alpha2, rho, mu and nu from alpha1, in ONE launch. The solver always wants all
// four -- alphaEqnSubCycle.H:36 rebuilds rho and interfaceProperties needs mu and nu right after -- so
// splitting them into four kernels would read alpha1 four times for no reason.
//
// Every pointer is device memory; `alpha2`, `rho`, `mu` and `nu` may alias nothing and are written in
// full. Any of the four outputs may be null and is then not computed.
void deviceMixtureCorrect(
    const scalar*               alpha1,
    int                         nCells,
    const DevicePhaseProperties& props,
    scalar*                     alpha2,
    scalar*                     rho,
    scalar*                     mu,
    scalar*                     nu);

// rho ON THE BOUNDARY, where alpha2's patch values are a field of their own and NOT 1 - alpha1's.
// `alpha2 = 1.0 - alpha1` (alphaEqn.H:223) runs one line above the corrector's mixture.correct(), and
// at a contact-angle wall that call rewrites alpha1's gradient and re-evaluates its patch; `rho ==
// alpha1*rho1 + alpha2*rho2` then blends the NEW alpha1 patch value with the OLD alpha2 one. Measured on
// capillaryRise against OpenFOAM's own rho*nu at the wall: 7.4e-05 out with 1 - alpha1, 1e-17 with this.
void deviceBoundaryRho(
    const scalar* alpha1Bnd,
    const scalar* alpha2Bnd,
    int nFaces,
    const DevicePhaseProperties& props,
    scalar* rhoBnd);

} // namespace brae
