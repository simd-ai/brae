#pragma once
// The rarefied-gas walls of rhoCentralFoam (applications/solvers/compressible/rhoCentralFoam/BCs, built into
// librhoCentralFoam and nowhere else in OpenFOAM v2412): maxwellSlipU on U and smoluchowskiJumpT on T, which
// biconic25-55Run35 runs on its cone (milestone I). CUDA arm only.
//
// provenance:
//   openfoam: BCs/U/maxwellSlipUFvPatchVectorField.C:120-170 (updateCoeffs), :77-117 (the dictionary constructor)
//             BCs/T/smoluchowskiJumpTFvPatchScalarField.C:122-162 (updateCoeffs), :76-115
//             src/finiteVolume/fields/fvPatchFields/derived/partialSlip/partialSlipFvPatchField.C:90-135 (snGrad,
//             evaluate, snGradTransformDiag), transformFvPatchField.C (gradientInternalCoeffs = -deltaCoeffs*
//             snGradTransformDiag, gradientBoundaryCoeffs = snGrad - cmptMultiply(gradientInternalCoeffs, pif))
//             src/thermophysicalModels/basic/derivedFvPatchFields/mixedEnergy/mixedEnergyFvPatchScalarField.C:96-125
//             src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/gaussGrad.C:135-160 (the patch value of fvc::grad(T))
//
// WHERE THEY UPDATE (rhoCentralFoam.C). maxwellSlipU's updateCoeffs runs at U.correctBoundaryConditions() after the
// explicit momentum (:232) and again at the viscous U matrix's construction (fvMatrix.C's psi.boundaryFieldRef().
// updateCoeffs()), on the same mu_b, rho_b, psi_b, T_b, grad(T) and tauMC: the same coefficients, computed once here.
// After the solve, U.correctBoundaryConditions() only evaluates. smoluchowskiJumpT's runs inside mixedEnergy's
// updateCoeffs (Tw.evaluate()), at e.correctBoundaryConditions() (:265) and at the e matrix's construction, after the
// first thermo.correct(): new mu_b and psi_b, the old rho_b. After the e solve the patch is updated(): T_b stays the
// construction's, and thermo.correct() takes e_b = he(p_b, T_b) from it (mixed fixesValue()).
//
// HOW brae CARRIES THEM. brae's boundary descriptors are per component and mixed (bcType 5): valueFraction, refValue,
// refGrad. smoluchowskiJumpT IS a mixed patch (refValue Twall, refGrad 0). maxwellSlipU is a partialSlip, a transform
// patch coupling U's components, and is carried the way brae carries slip and wedge: each component a mixed face with
// valueFraction_k = snGradTransformDiag_k = vf + (1 - vf)*|n_k| and refValue_k such that the mixed gradient
// coefficients are partialSlip's, -deltaCoeffs*diag_k and deltaCoeffs*(U_b,k - U_c,k + diag_k*U_c,k); that refValue
// depends on the cells' U, so it is rebuilt from the U at hand before each use (emulate). The patch VALUE is written
// as partialSlip::evaluate computes it, lerp(transform(I - n n, U_c), refValue, vf) (evaluate).
#include "cf_types.cuh"
#include "device_boundary.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "rhoCentralControls.cuh"
#include "rhoCentralState.cuh"

namespace brae {
namespace gpu {
namespace rhoCentral {

using brae::rhoCentral::Case;

struct DeviceFields;

class SlipWalls
{
public:
    SlipWalls(const Case& cs, const DeviceMesh& dm, DeviceVectorBoundary& dbU, DeviceBoundary& dbT, int control = 0);

    bool maxwell() const { return nMs_ > 0; }
    bool smoluchowski() const { return nSj_ > 0; }

    // maxwellSlipU::updateCoeffs: valueFraction and refValue from mu(T_b), rho_b, psi_b, T_b, fvc::grad(T) (its patch
    // value) and tauMC_b (packed [(i*3+j)*nB + k], rhoCentralViscous's tauB)
    void updateMaxwell(const DeviceBuffer<scalar>& T, const DeviceBuffer<scalar>& TB, const DeviceBuffer<scalar>& rhoB,
                       const DeviceBuffer<scalar>& psiB, const DeviceBuffer<scalar>& tauB);
    // partialSlip in brae's per-component mixed form, from the cells' U at hand
    void emulate(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                 const DeviceBuffer<scalar>& Uz);
    // partialSlip::evaluate into U_b
    void evaluate(const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                  DeviceBuffer<scalar>& UxB, DeviceBuffer<scalar>& UyB, DeviceBuffer<scalar>& UzB);

    // smoluchowskiJumpT::updateCoeffs on T's descriptor (valueFraction; refValue Twall and refGrad 0 are set once),
    // and mixedEnergy's refValue he(p_b, Twall) for e's matrix
    void updateSmoluchowski(const DeviceBuffer<scalar>& TB, const DeviceBuffer<scalar>& rhoB,
                            const DeviceBuffer<scalar>& psiB, const DeviceBuffer<scalar>& pB, DeviceBoundary& dbT);
    // T_b as the e matrix's construction evaluated it, kept through the solve's correctBoundaryConditions()
    void keepTb(const DeviceBuffer<scalar>& TbEval);
    void restoreTb(DeviceBuffer<scalar>& TbEval) const;
    // e's mixedEnergy refValue on every boundary face (0 off the smoluchowskiJumpT faces)
    const DeviceBuffer<scalar>& eRef() const { return eRef_; }

private:
    const Case& cs_;
    const DeviceMesh& dm_;
    int nMs_ = 0, nSj_ = 0;
    int noJump_ = 0;   // the negative control smoluchowskiNoJump
    DeviceBuffer<label> msFaces_, msCell_, sjFaces_;
    DeviceBuffer<scalar> msN_[3], msDc_, msAcc_, msUw_[3], msVf_, msR_[3];
    DeviceBuffer<int> msCreep_, msCurv_;
    DeviceBuffer<scalar> gx_, gy_, gz_;   // fvc::grad(T) on the cells
    DeviceBuffer<scalar> sjDc_, sjAcc_, sjTw_, sjGamma_, sjTb_, eRef_;
};

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
