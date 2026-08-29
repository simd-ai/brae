#pragma once
// CUDA implementation of rhoSimpleFoam's momentum predictor -- the device twin of rhoUEqn_cpp.
//
// provenance:
//   openfoam:  applications/solvers/compressible/rhoSimpleFoam/UEqn.H:1-21
//     also:    src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress.C:107-117
//              (divDevRhoReff -- the COMPRESSIBLE overload, taking rho)
//   reference: src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu   (gated against OpenFOAM's own dumps)
//   cuda:      src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu
//   tests:     tests/test_rho_ueqn_cuda.cu -- field by field against the _cpp reference: diag, upper,
//              lower, source[0..2] and iC[k]/bC[k] on every patch, never folded into one number.
//              Registered as seven cases (laminar, turbulent, and one per div scheme). Measured on
//              matrixDumpAsym/282 and pitzDailyTurb/1576: diag 8.700e-12 / 2.018e-12, off-diagonals
//              1.393e-11 / 3.962e-12, and every boundary coefficient at 0.000e+00 on both.
//              THE CONTROL IS THE ONE THE MANIFEST NAMES: injecting the KINEMATIC nuEff -- the
//              incompressible divDevReff -- moves the diagonal 2.883e-01, and removing the rho weighting
//              from this module turns the gate red at 2.883e-01, the number the control predicted.
//
// WHY THE NAME IS rhoUEqn AND NOT UEqn. brae puts every source directory on ONE include path, so a
// `UEqn.cuh` here would resolve to src/applications/solvers/simpleFoam/UEqn.cuh, or the reverse,
// depending on which the compiler saw first. simpleFoam already owns UEqn.cuh and pEqn.cuh. The host
// references are rhoUEqn_cpp / rhoPEqn_cpp / rhoEEqn_cpp for the same reason and these follow them.
//
// THE CONTRACT WITH THE REFERENCE, unchanged from the incompressible twin: this produces the SAME OBJECT
// rhoUEqn_cpp produces, in the same decomposition -- one shared scalar LDU (diag/upper/lower), a
// per-component source, per-component boundary coefficients -- and it does NOT fold the boundary into the
// diagonal or fuse the stages. A fused kernel can only be compared as one number, and the whole method
// here is to name the first divergent stage.
//
// WHAT MAKES THIS DIFFERENT FROM simpleFoam/UEqn.cu, and it is one factor:
//
//     divDevRhoReff(U) = -fvc::div((alpha*rho*nuEff)*dev2(T(grad U))) - fvm::laplacian(alpha*rho*nuEff, U)
//     divDevReff(U)      the INCOMPRESSIBLE lineage, where alpha == 1 and rho == 1
//
// so this equation carries the DYNAMIC viscosity mu_eff = rho*nu_eff where the incompressible one carries
// the KINEMATIC nu_eff. That factor is order 1 for air at ambient conditions and order 10 across a
// compressible duct, so a port that reuses the incompressible form is wrong by an amount that LOOKS
// plausible and varies with the solution. It cannot be caught by inspecting a converged field. The host
// gate measures it: assembling with the kinematic nu_eff reads 6.2e-01 against OpenFOAM's own matrix,
// fourteen orders worse than the dynamic form's 6.1e-15.
//
// rho AND nuEff ARE TAKEN SEPARATELY, NOT AS A PRE-MADE muEff -- the same decision the reference records.
// Handing this module a finished muEff would move the one product that distinguishes the solver out of
// the file that documents it. The gate's injection path (muEffCell/muEffBndFace below) is the deliberate
// exception, and it exists so the assembly can be measured without a ported compressible closure in the
// way, exactly as on the host.
//
// phi IS THE MASS FLUX (kg/s), from compressibleCreatePhi.H -- not the volumetric flux. fvm::div(phi,U)
// is therefore already rho-weighted and the operator needs no change; the field it is given is what
// differs. Passing a volumetric phi is the same class of error as passing nuEff for muEff.
//
// THE FACE VISCOSITY IS NOT THE OWNER CELL'S. effectiveFaceViscosity interpolates muEff to the faces and
// then overwrites the BOUNDARY faces with the patch values, because on a wall carrying a nut wall
// function the face value differs from the owner cell's by the whole of nut_wall. The reference makes the
// boundary array a required argument for that reason and so does this.
//
// `corrected` HAS TWO HALVES AND BOTH ARE REQUIRED. correctedLaplacian switches the implicit coefficient
// to nonOrthDeltaCoeffs AND adds the explicit deferred source
// -V*div(gamma*magSf*(corrVec & interpolate(grad U))). Implementing only the implicit half is a defect
// this port has already paid for on the ENERGY and PRESSURE equations, where it left the source short by
// the whole correction while the diagonal stayed exact -- so every gate that compared D() passed. See
// PORT.md. DeviceMesh already carries nonOrthDc and corrVec for both halves.
//
// WHAT THE CALLER MUST HAVE DONE BEFORE ENTERING, and it is not advisory. The host reference is handed
// live fvPatchField objects, so fvm::div and fvm::laplacian read valueInternalCoeffs/gradientInternalCoeffs
// off the CURRENT state -- OpenFOAM's fvMatrix constructor runs updateCoeffs() at every assembly. This
// module is handed a PRE-BAKED DeviceVectorBoundary whose bcType, refValue, valueFraction and symMask are
// a snapshot, and it refreshes none of them. Every equivalent lives outside this file, and the relax step
// below reasons from symMask being the per-component valueFraction, which only holds if the symmetry
// refresh ran against THIS iteration's U. In order, before assembleUEqn:
//
//   1. MRF.correctBoundaryVelocity(U) -- UEqn.H:3, on the HOST field, BEFORE buildDeviceVectorBoundary,
//      because the included patch faces take the frame velocity Omega x (Cf - origin) and that value has
//      to be in the field the snapshot is taken from (cpu::MRF::correctBoundaryVelocity;
//      simpleFoamV2.cu:652-674 is the worked example). This module applies MRF.DDt(rho, U) and NOT this.
//   2. deviceUpdateInletOutlet(dbU, phiBnd) -- the io switch, with the PREVIOUS step's boundary flux, as
//      OpenFOAM lags it.
//   3. deviceUpdatePressureInletOutletVelocity, deviceUpdateSymmetry, deviceUpdateWedge,
//      deviceUpdateMixedFreestream -- after the io switch, against this iteration's cell velocity.
//   4. deviceUpdateFlowRateInlet -- and on THIS solver, note which rho: OpenFOAM recomputes avgU from the
//      LIVE rho at every updateCoeffs. Feeding it thermo.rho() while the flux is built from the relaxed
//      solver rho is the angledDuct defect, and this module cannot see it.
//
// WHICH rho. rhoCell/rhoBndFace are the SOLVER's volScalarField rho -- the one createFields.H:44-53 hands
// compressible::turbulenceModel::New, so `this->rho_` inside divDevRhoReff IS that field, and the one
// UEqn.H:8 passes to MRF.DDt. pEqn.H:105-109 sets it to thermo.rho() and then RELAXES it (`rho.relax()`
// unless transonic), so it is NOT thermo.rho(). The two differ by O(1 - alpha_rho) of the density change
// per iteration: small, plausible, solution-dependent, and invisible in a converged field -- the same
// shape of error as the kinematic-for-dynamic viscosity one above.
//
// REFUSED, not ignored -- identical to the reference: MRF, and any fvOptions type that is not ported.
// REFUSED here and NOT in the reference: a mesh with a coupled patch, and a phi of the wrong length.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_MRF.cuh"
#include "device_fvoptions.cuh"   // DevicePorosity
// MomentumMatrix and addPressureGradient. The assembled object is the SAME in both lineages -- as
// FvMatrix<vector> is on the host -- so it is reused rather than defined a second time: two structs of
// the same name and shape is the hazard the include-path note above describes. It lives in the
// incompressible solver's header for historical reasons and belongs in a shared one the moment a third
// lineage needs it.
#include "UEqn.cuh"
// cpu::rhoSimple::DivScheme. The compressible lineage keeps its OWN enum rather than sharing the
// incompressible one, so that adding a scheme here is a decision taken against rhoSimpleFoam's own
// tutorials instead of inherited silently. That is the reference's decision and this follows it.
#include "rhoUEqn_cpp.cuh"
#include <string>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoSimple {

struct RhoMomentumInput
{
    // MASS flux, kg/s. NOT the volumetric flux -- see the header note.
    const DeviceBuffer<scalar>* phiInt = nullptr;        // internal faces
    const DeviceBuffer<scalar>* phiBnd = nullptr;        // boundary faces

    // rho and the KINEMATIC nuEff, kept separate so that mu_eff = rho*nu_eff is formed HERE.
    const DeviceBuffer<scalar>* rhoCell     = nullptr;   // nCells
    const DeviceBuffer<scalar>* rhoBndFace  = nullptr;   // boundary faces, the PATCH value
    const DeviceBuffer<scalar>* nuEffCell   = nullptr;   // nCells
    const DeviceBuffer<scalar>* nuEffBndFace = nullptr;  // boundary faces, the PATCH value

    // ALTERNATIVE to rhoCell/nuEffCell: the dynamic viscosity directly. When set these are used verbatim
    // and rho/nuEff are ignored. This exists for the gate, which injects OpenFOAM's own
    // turbulence->muEff() so the ASSEMBLY can be measured without a ported compressible closure in the
    // way -- the same escape hatch, for the same reason, as the host reference's muEff/muEffBnd.
    const DeviceBuffer<scalar>* muEffCell    = nullptr;  // nCells
    const DeviceBuffer<scalar>* muEffBndFace = nullptr;  // boundary faces

    // UEqn.relax(). TWO fields, because OpenFOAM's guard is on the PRESENCE of the entry, not its value:
    // fvMatrix::relax() runs `if (relaxEquation(name, relaxCoeff)) relax(relaxCoeff)` (fvMatrix.C:1250-1263)
    // and relax(alpha) early-returns only at alpha <= 0 (fvMatrix.C:1102-1107). So `equations { U 1; }` --
    // an ordinary SIMPLEC setting -- DOES relax, applying the diagonal-dominance clamp D = max(|D|,sumOff),
    // the asymmetric boundary add/remove and S += (D - D0)*psi, none of which is the identity. relaxU == 1
    // must therefore not be used as the sentinel for "the case named no factor".
    //
    // relaxEquationU: did fvSolution's relaxationFactors/equations resolve "U" (or "default")? The caller
    // reads that; nothing here can. FALSE means no relaxation at all, whatever relaxU holds.
    bool   relaxEquationU = false;
    scalar relaxU = 1.0;
    // `bounded Gauss <scheme>`: diag -= V*div(phi). Vanishes at convergence, unlike linearUpwind's.
    bool   bounded = false;
    // `Gauss linearUpwind grad(U)`: the matrix stays pure upwind and the whole scheme is a deferred
    // source correction. It does NOT vanish at convergence.
    bool   linearUpwind = false;
    // The `k` of `grad(U) cellLimited Gauss linear <k>` -- the gradient linearUpwind NAMES. 0 leaves the
    // plain Gauss gradient. An unlimited gradient under a limited name is a different equation.
    scalar gradULimitK = 0.0;
    cpu::rhoSimple::DivScheme scheme = cpu::rhoSimple::DivScheme::upwind;
    scalar schemeCoeff = 1.0;            // the `k` of `limitedLinear k`
    bool   correctedLaplacian = false;   // both halves -- see the header note
    scalar snGradLimitCoeff = 0.0;       // `limited <k> corrected`; 0 = uncapped, which is what `corrected` means

    // Declared by the case but not ported -> REFUSE. A solver that read MRFProperties, ignored it,
    // converged and reported nothing wrong has already shipped here once.
    bool hasMRF = false;
    bool hasFvOptions = false;
    // Does the mesh carry a cyclic / cyclicAMI / cyclicACMI / processor patch? DeviceMesh cannot say --
    // buildDeviceMesh SKIPS coupled patches in the boundary gather and keeps them out of the internal-face
    // LDU by design (device_mesh.cuh:34-38, 110-123) -- so the DRIVER says it, from isCoupledInterfaceType
    // over the patch list, exactly as it already says hasMRF. This assembly passes no interface to
    // divDevRhoReff, to the corrected-laplacian source, to the off-diagonals or to the relaxation's
    // diagonal-dominance sum, so with a coupled patch present all four drop those faces silently. Set it
    // and this module REFUSES; a comment is not a refusal.
    bool hasCoupledPatches = false;
    // WHICH option, so the refusal names it. A generic "the case declares fvOptions" sends the reader
    // back to the dictionary to work out what to implement next.
    std::string fvOptionUnsupported;
    // The IMPLEMENTED ones. UEqn.H applies fvOptions(rho, U); for explicitPorositySource that is
    // eqn -= porosityEqn with the resistance built by the porosity model. Null = no options to apply.
    // NOTE the compressible equation is FORCE-dimensioned, so fixedCoeff reads rhoRef from the dict and
    // DarcyForchheimer takes the rho branch -- both models dispatch on UEqn.dimensions() in OpenFOAM.
    const DevicePorosity* porosity = nullptr;
    // MRF.DDt(rho, U) == rho*DDt(U), UEqn.H:8 -- zones resolved against the mesh and uploaded.
    const std::vector<DeviceMRFZone>* mrf = nullptr;
};

// UEqn.H steps 1-2: fvm::div(phi,U) + MRF.DDt(rho,U) + turbulence->divDevRhoReff(U), then UEqn.relax().
//
// Returns the relaxed momentum matrix WITHOUT the pressure gradient, because that is the object pEqn.H
// needs for A() and H(). Throws (host-side) on MRF or an unported fvOptions, matching the reference's
// refusal contract exactly -- the gate asserts the refusals as well as the numbers.
void assembleUEqn(
    MomentumMatrix&             M,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const RhoMomentumInput&     in);

// UEqn.H step 3 -- solve(UEqn == -fvc::grad(p)) -- is gpu::addPressureGradient (UEqn.cuh), unchanged:
// source -= grad(p)*V is the same operation in both lineages and is applied by the driver to a COPY,
// because pEqn.H needs the original matrix. It is not redeclared here.

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
