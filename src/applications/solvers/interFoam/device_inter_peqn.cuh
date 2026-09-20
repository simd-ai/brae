#pragma once
// interFoam's pressure corrector on the device -- the four things that are interFoam's OWN.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/interFoam/pEqn.H:1-89
//              src/finiteVolume/finiteVolume/ddtSchemes/EulerDdtScheme/EulerDdtScheme.C (fvcDdtPhiCorr)
//              src/finiteVolume/finiteVolume/ddtSchemes/ddtScheme/ddtScheme.C (fvcDdtPhiCoeff)
//   host:      src/applications/solvers/interFoam/inter_peqn_cpp.cu -- the ORACLE, gated in
//              tests/test_inter_peqn_cpp.cu and running end to end at 2.3e-06 relative in p_rgh on
//              damBreak against real OpenFOAM.
//   tests:     tests/test_device_inter_peqn.cu
//
// The laplacian, the solve and the non-orthogonal loop are machinery brae already has and shares with
// every other pressure corrector. What is ported here is the four places a port that copies simpleFoam's
// pEqn -- or interFoam's own UEqn -- is wrong, and each still converges when it is:
//
//   1. phig CARRIES NO snGrad(p_rgh), where UEqn's source does. The pressure gradient is EXPLICIT in
//      the momentum predictor and IMPLICIT here -- it is the laplacian being solved. Carrying it into
//      phig counts it twice and converges to a flow with the wrong balance at the interface.
//      deviceBuoyancyFlux takes no p_rgh at all, so the term cannot be passed by accident.
//
//   2. ddtCorr IS WEIGHTED BY interpolate(rho*rAU), NOT interpolate(rho)*rAUf. The PRODUCT is formed
//      per cell and interpolated once. Linear interpolation does not commute with multiplication, and
//      the gap is largest where the two factors vary most -- across a VoF interface that is a factor of
//      1000 in a single face.
//
//   3. THE VELOCITY CORRECTION DIVIDES BY rAUf INSIDE reconstruct AND MULTIPLIES BY rAU OUTSIDE:
//          U = HbyA + rAU*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf)
//      The two forms coincide EXACTLY when rAU is uniform -- which it is on any single-phase fixture
//      with a uniform mesh -- so a gate built on one cannot tell them apart.
//
//   4. THE ddtCorr COEFFICIENT IS A LIMITER BY DEFAULT, NOT A CONSTANT. ddtPhiCoeff_ is -1 unless
//      fvSchemes says otherwise, selecting 1 - min(|phiCorr|/(|phi| + SMALL), 1) -- the correction is
//      switched OFF where it is large compared with the flux itself. A constant 1 applies it hardest
//      where OpenFOAM applies it least. It is also zeroed on every patch where U fixes a value.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_MRF.cuh"
#include "device_cyclic.cuh"

namespace brae {

// phig = (surfaceTensionForce - ghf*snGrad(rho)) * rAUf * magSf, pEqn.H:28-34. No p_rgh -- see 1.
// `magSf` and the three face fields are the mesh's FULL face arrays, internal faces first.
void deviceBuoyancyFlux(
    int                         n,
    const DeviceBuffer<scalar>& surfaceTensionForce,
    const DeviceBuffer<scalar>& ghf,
    const DeviceBuffer<scalar>& snGradRho,
    const DeviceBuffer<scalar>& rAUf,
    const DeviceBuffer<scalar>& magSf,
    DeviceBuffer<scalar>&       phig);

// fvc::interpolate(rho*rAU) on the internal faces -- the product per cell, interpolated once. See 2.
void deviceRhoRAUf(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& rAU,
    DeviceBuffer<scalar>&       out);

// fvc::ddtCorr(U, phi), Euler, fixed mesh. `bndUFixesValue` is 1 on every boundary face whose patch
// fixes U's value, and the correction is zero there -- see 4. It is passed rather than derived because
// which patches fix a value is a fact about boundary conditions, and the device mesh carries none.
void deviceDdtCorr(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& phiOldInt,
    const DeviceBuffer<scalar>& phiOldBnd,
    const DeviceBuffer<scalar>& UOldX,
    const DeviceBuffer<scalar>& UOldY,
    const DeviceBuffer<scalar>& UOldZ,
    const DeviceBuffer<int>&    bndUFixesValue,
    scalar                      ddtPhiCoeff,      // negative selects the limiter -- the default
    scalar                      deltaT,
    DeviceBuffer<scalar>&       outInt,
    DeviceBuffer<scalar>&       outBnd,
    // U.oldTime()'s STORED PATCH values, which fvc::dotInterpolate uses on an uncoupled patch
    // (surfaceInterpolationScheme.C:296-298). Null falls back to the face cell, which is the same only
    // where the patch value follows the cell -- not on a slip wall.
    const DeviceBuffer<scalar>* UOldBndX = nullptr,
    const DeviceBuffer<scalar>* UOldBndY = nullptr,
    const DeviceBuffer<scalar>* UOldBndZ = nullptr);

// U = HbyA + rAU*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf), pEqn.H:58. `faceFlux` is the
// difference BEFORE the division; the division happens inside so the two operations cannot be
// separated at a call site and quietly reordered -- see 3.
void deviceCorrectVelocity(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& HbyAX,
    const DeviceBuffer<scalar>& HbyAY,
    const DeviceBuffer<scalar>& HbyAZ,
    const DeviceBuffer<scalar>& rAU,
    const DeviceBuffer<scalar>& faceFluxInt,
    const DeviceBuffer<scalar>& rAUfInt,
    const DeviceBuffer<scalar>& faceFluxBnd,
    const DeviceBuffer<scalar>& rAUfBnd,
    DeviceBuffer<scalar>&       UX,
    DeviceBuffer<scalar>&       UY,
    DeviceBuffer<scalar>&       UZ,
    // ...and the PERIODIC PAIR's faces, which surfaceSum walks with the rest of mesh.boundary()
    // (fvcSurfaceIntegrate.C:168-180). `faceFluxIf` is phig - p_rghEqn.flux() there and `rAUfIf` the
    // interpolated rAU, exactly as the two boundary arrays above are. Null on a mesh without a pair.
    const DeviceCyclic*         cyc         = nullptr,
    const DeviceBuffer<scalar>* faceFluxIf  = nullptr,
    const DeviceBuffer<scalar>* rAUfIf      = nullptr);

// p = p_rgh + rho*gh, pEqn.H:72 -- the field interFoam writes and never solves.
void deviceStaticPressure(
    int                         nC,
    const DeviceBuffer<scalar>& p_rgh,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& gh,
    DeviceBuffer<scalar>&       p);

// fvc::interpolate(vf) over the mesh's FULL face array: the linear weights on the internal faces, and
// the FACE CELL's value at an uncoupled patch -- which is what surfaceInterpolation gives there, since
// there is no second cell to weight against. rAUf is wanted in exactly that form by the pressure step,
// which reads its head for the laplacian and its tail for the velocity correction.
void deviceInterpolateFull(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& vf,
    DeviceBuffer<scalar>&       out);

// A cell field gathered to the boundary faces -- the face cell's value, one per boundary face. This is
// zeroGradient, and it is what rho's patch values are when no patch condition says otherwise.
void deviceGatherBoundary(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& vf,
    DeviceBuffer<scalar>&       out);

// ---------------------------------------------------------------------------------------------------
// THE p_rgh MATRIX: fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA), pEqn.H:44-56.
//
// `iC`/`bC` are p_rgh's own boundary coefficients, flattened in boundary-face order, from the host's
// per-patch valueInternalCoeffs/gradientInternalCoeffs -- fixedFluxPressure, totalPressure and
// zeroGradient each contribute differently and that dispatch stays on the host, as everywhere else in
// this port. Everything that scales with the CELL COUNT is here.
//
// THE SIGN. `pe.source[c] += div[c]*V[c]` -- fvMatrix::operator== is source += V*R, a PLUS, where
// rhoSimpleFoam's momentum path carries the minus inside R = -grad(p). Both conventions live in this
// tree and getting this one backwards still converges, to a pressure field that drives the flow the
// wrong way.
//
// setReference PINS ONE CELL when no patch fixes a value: source += diag*refValue AND diag += diag --
// it DOUBLES the diagonal entry rather than replacing the row. damBreak does not need it (its
// atmosphere patch is totalPressure) but 8 of the shipped tutorials have no value-fixing p_rgh patch
// at all and every one of those does.
struct DevicePressureMatrix
{
    DeviceBuffer<scalar> diag, upper, lower;   // the RAW LDU, before the boundary fold
    DeviceBuffer<scalar> source;               // extensive
};

// phiHbyA's two interFoam-only terms, added to the flux of HbyA that the shared pressure predictor
// already built (pEqn.H:36-42):
//
//     phiHbyA += fvc::interpolate(rho*rAU) * fvc::ddtCorr(U, phi)      internal faces only
//     phiHbyA += phig                                                  internal faces AND boundary
//
// THE BOUNDARY HALF OF phig IS NOT OPTIONAL. `phiHbyA += phig` in pEqn.H is a whole-surfaceScalarField
// operation and fvc::div(phiHbyA) sums the boundary faces, so a wall's buoyancy and surface tension
// enter the PRESSURE EQUATION'S SOURCE through it. Adding phig to the internal faces alone drops that
// entirely -- and at a contact-angle wall it is the term the contact angle exists to apply. On
// capillaryRise, where momentumPredictor is off, the pressure corrector is the ONLY route surface
// tension has into the solution at all.
//
// ddtCorr HAS NO BOUNDARY HALF HERE, AND OpenFOAM'S DOES. This comment used to say fvc::interpolate
// builds interpolate(rho*rAU) on the internal faces only; it builds the patch values too, and
// fvcDdtPhiCoeff zeroes the coupling coefficient only where U FIXES A VALUE (ddtScheme.C). On a patch
// that does not -- RAS/weirOverflow's `U zeroGradient` outlet -- the correction is live, and the host
// reference carries it since that case's gate found it missing (inter_peqn_cpp.cu). The device loop
// REFUSES a case with such an open patch (inter_driver_device.cu) until this kernel is taught the same.
void deviceInterAddPhiHbyATerms(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rhoRAUfInt,     // interpolate(rho*rAU)
    const DeviceBuffer<scalar>& ddtCorrInt,
    const DeviceBuffer<scalar>& phigInt,
    const DeviceBuffer<scalar>& phigBnd,
    bool                        haveDdtCorr,    // false on a start from rest
    DeviceBuffer<scalar>&       phiHbyAInt,
    DeviceBuffer<scalar>&       phiHbyABnd,
    // MRFZoneList::makeRelative(phiHbyA), pEqn.H:19 -- BETWEEN the ddtCorr term and phig, which is why
    // it is applied in here rather than by the caller
    const std::vector<DeviceMRFZone>* mrf = nullptr,
    // ddtCorr's BOUNDARY half and what interpolate(rho*rAU) needs for it: rho's patch values and rAU on
    // cells (inter_peqn_cpp.cu:531-573). Null = the term is not added, which is right only where every
    // open patch fixes U.
    const DeviceBuffer<scalar>* ddtCorrBnd = nullptr,
    const DeviceBuffer<scalar>* rhoBnd = nullptr,
    const DeviceBuffer<scalar>* rAU = nullptr,
    // ...and U's per-face fixesValue mask, because the host skips such a patch outright
    const DeviceBuffer<int>*    uFixesValue = nullptr);

// phi = phiHbyA - p_rghEqn.flux(), pEqn.H:56. fvMatrix::flux() is
//     internal  upper*p[nei] - lower*p[own]
//     boundary  internalCoeffs*p[faceCell] - boundaryCoeffs      <- the face CELL's value, not the patch's
// and `phi = phiHbyA - flux` is what makes phi conservative. The flux is ALSO what the velocity
// correction reads, so it is returned rather than folded away.
// `faceFluxCorrection`: the corrected laplacian's non-orthogonal flux on the internal faces, which
// fvMatrix::flux() adds (fvMatrix.C:1688, the host's matrixFlux); null under an orthogonal assembly
void deviceInterPEqnFlux(
    const DeviceMesh&           dm,
    const DevicePressureMatrix& P,
    const DeviceBuffer<scalar>& iC,
    const DeviceBuffer<scalar>& bC,
    const DeviceBuffer<scalar>& pSolved,
    DeviceBuffer<scalar>&       fluxInt,
    DeviceBuffer<scalar>&       fluxBnd,
    const DeviceBuffer<scalar>* faceFluxCorrection = nullptr);

// `corrected`: the case's `Gauss linear corrected` -- nonOrthDeltaCoeffs on the internal faces -- and
// `nonOrthSource`, deviceFaceDivSource of the correction flux (the host's laplacianNonOrthSource with its
// sign flipped), added before div(phiHbyA) and the reference as the host's pressureCorrector does; null
// under an orthogonal assembly
void deviceInterAssemblePEqn(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rAUfInt,       // fvc::interpolate(rAU) on the internal faces
    const DeviceBuffer<scalar>& phiHbyAInt,
    const DeviceBuffer<scalar>& phiHbyABnd,
    bool                        needReference,
    int                         pRefCell,
    // p_rgh itself: setReference pins the cell at its CURRENT value (pEqn.H:47), not at pRefValue
    const DeviceBuffer<scalar>* pRghForRef,
    DevicePressureMatrix&       P,
    bool                        corrected = false,
    const DeviceBuffer<scalar>* nonOrthSource = nullptr,
    // the periodic pair, and rAU as a CELL field for the interpolation the interface does itself
    DeviceCyclic*               cyc = nullptr,
    const DeviceBuffer<scalar>* rAUCell = nullptr,
    // the pair's phiHbyA: fvc::div(phiHbyA) is the pressure equation's SOURCE and sums a coupled face
    const DeviceBuffer<scalar>* phiHbyAIf = nullptr);

// pEqn.H:74-83, after p = p_rgh + rho*gh: shift p so that p[pRefCell] is pRefValue, and REBUILD p_rgh
// from the shifted p (applyPressureReference, inter_peqn_cpp.cu:179-196). Both fields move.
void deviceInterPressureReference(
    int                         nC,
    int                         pRefCell,
    scalar                      pRefValue,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& gh,
    DeviceBuffer<scalar>&       p,
    DeviceBuffer<scalar>&       p_rgh);

} // namespace brae
