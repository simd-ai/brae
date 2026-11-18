#pragma once
// rhoCentralViscous.cuh -- rhoCentralFoam's viscous correctors on the device arm (milestone G).
//
// OpenFOAM v2412 rhoCentralFoam.C, when max(thermo.mu()) > 0 (createFieldRefs.H), in the step's order:
//   :218-219  tauMC = muEff*dev2(T(fvc::grad(U)))            from U as the step starts, cells and patch faces
//   :222-232  (the caller) rho and rhoU explicit, U = rhoU/rho, U.correctBoundaryConditions()
//   :234-244  solve(fvm::ddt(rho, U) - fvc::ddt(rho, U) - fvm::laplacian(muEff, U) - fvc::div(tauMC)); rhoU = rho*U
//   :246-253  sigmaDotU = (interpolate(muEff)*magSf*snGrad(U) + dotInterpolate(Sf, tauMC)) & (a_pos*U_pos + a_neg*U_neg)
//   :258-271  (the caller) rhoE explicit with - fvc::div(sigmaDotU); e, e.correctBoundaryConditions(), thermo.correct()
//   :273-282  solve(fvm::ddt(rho, e) - fvc::ddt(rho, e) - fvm::laplacian(alphaEff, e)); (the caller) thermo.correct(),
//             rhoE = rho*(e + 0.5*magSqr(U))
//
// Built from brae's shared pieces, none copied: deviceGaussGradFused (grad(U), grad(e)), deviceBoundaryGradU
// (gaussGrad's patch correction), deviceScaledDev2T (muEff*dev2(T(.))), deviceDotInterpolateSf, deviceDiv
// (surfaceIntegrate), deviceLaplacianGammaMagSf (+ deviceLaplacianCorrFlux / deviceFaceDivSource, the corrected
// scheme's explicit part), deviceEulerFvmDdtRho / deviceEulerFvcDdtRhoSub, deviceBCLaplacianCoeffsFace (U's patches),
// deviceFold, deviceNormFactorInto and deviceSymGaussSeidelFused (smoothSolver with OpenFOAM's sequential
// GaussSeidel sweep), deviceSnGradInternal. rhoCentralFoam's own: the order above, e's patch coefficients from T's
// (fixedEnergy, gradientEnergy, mixedEnergy, heThermo's heBoundaryTypes), and sigmaDotU's patch faces.
//
// constTransport: muEff and alphaEff are constant and the laplacian matrices are built once. sutherland (milestone H):
// muEff from the T the step starts from and alphaEff from the T of its first thermo.correct(), the laplacians rebuilt on
// their linear interpolates each step. Wedge patches: wedgeFvPatchField's snGrad and coefficients (cellT, half the
// deltaCoeffs), through the shared deviceUpdateWedge / deviceWedgeLaplacianBC.
#include "rhoCentralControls.cuh"
#include "rhoCentralFlux.cuh"
#include "rhoCentralState.cuh"

#include "device_boundary.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"

namespace brae {
namespace gpu {
namespace rhoCentral {

class ViscousCorrectors
{
public:
    ViscousCorrectors(const brae::rhoCentral::Case& cs, const DeviceMesh& dm, int control);

    // :218-219 and the old-time copies. F as the step starts; dbU's stored values are U's patch values.
    void beginStep(const DeviceFields& F, const DeviceVectorBoundary& dbU);
    // :234-244 on the U the explicit update left, its patches evaluated (the caller's U.correctBoundaryConditions).
    // Leaves U solved on the cells and rhoU = rho*U there; the caller evaluates U's patches again.
    void solveU(DeviceFields& F, DeviceVectorBoundary& dbU, scalar rDeltaT, brae::rhoCentral::StepReport& rep);
    // :246-253 then its fvc::div (over V), from the corrected U with its patches evaluated
    const DeviceBuffer<scalar>& divSigmaDotU(const DeviceFields& F, const DeviceVectorBoundary& dbU,
                                             const DeviceFluxes& X);
    // :273-277 on the e the explicit update left (its patches evaluated, thermo corrected). Leaves e solved on the
    // cells; the caller evaluates its patches, corrects the thermo and sets rhoE.
    void solveE(DeviceFields& F, const DeviceBoundary& dbT, scalar rDeltaT, brae::rhoCentral::StepReport& rep);
    // The step's viscous fluxes INTO the domain through the patches, times deltaT, added to inflow[0..3] (x, y, z
    // momentum, energy): U's laplacian boundary flux plus Sf & tauMC_b, and sigmaDotU_b plus e's. Diagnostics.
    void boundaryInflow(const DeviceFields& F, scalar deltaT, DeviceBuffer<double>& inflow);
    // What the implicit correctors leave unconserved: deltaT*sum over the cells of b - A x (lduMatrix::residual) after
    // each solve, summed over the run -- x, y, z momentum and energy. A converged solve leaves round-off; the
    // tutorials' relTol 0.01 leaves more, the same as OpenFOAM's algorithm does (the totals move by minus this).
    void solveResidual(double out[4]) const { for (int k = 0; k < 4; ++k) out[k] = solveRes_[k]; }
    // The same remainder unsigned, deltaT*sum|b - A x| (the actual, unnormalized algebraic residual), and the largest
    // error, over the run's solves, of the identity each solve satisfies (x the solved field, x_p the field before it):
    //   sum_c rho_c V_c (x_c - x_p,c) = deltaT*[sum_b (bC_b - iC_b x_c(b)) (+ sum_b Sf_b & tauMC_b for U)] - deltaT*sum_c r_c
    // relative to sum_c rho_c V_c |U_c| (momentum) or |e_c| (energy): the signs, the volume weights and the patch contributions checked one solve
    // at a time. x, y, z momentum and energy.
    void solveResidualL1(double out[4]) const { for (int k = 0; k < 4; ++k) out[k] = solveResL1_[k]; }
    void solveIdentity(double out[4]) const { for (int k = 0; k < 4; ++k) out[k] = identityMax_[k]; }

private:
    const brae::rhoCentral::Case& cs_;
    const DeviceMesh& dm_;
    int control_;
    int nC_, nIF_, nB_;
    int valid_[3];
    bool nonOrth_ = false;   // some nonOrthCorrectionVector is not exactly zero: the corrected schemes' explicit parts
    scalar mu_, alphaEff_;

    DeviceBuffer<scalar> muC_, muB_, muF_, alphaF_, negMuB_, negAlphaB_, ones_;
    DeviceBuffer<scalar> lapDiagU_, negUpperU_, lapDiagE_, negUpperE_;
    DeviceBuffer<scalar> rho0_, U0_[3], e0_;
    DeviceBuffer<scalar> gx_[3], gy_[3], gz_[3], gradU_, gradB_, tauC_, tauB_;
    DeviceBuffer<scalar> dTau_[3], dTauB_[3], divTau_[3];
    DeviceBuffer<scalar> diagD_, diagM_, src_[3], lapCorr_, ffc_, iC_[3], bC_[3], diagC_[3], b_[3], dnf_[3];
    DeviceLduView A_[3];
    DeviceBuffer<scalar> snG_[3], snGB_[3], sigma_, sigmaB_, divSigma_;
    DeviceBuffer<scalar> ge_[3], srcE_, diagE_, iCe_, bCe_, diagCE_, bE_, dnfE_;
    DeviceLduView AE_;
    DeviceBuffer<int> uKind_, Tconstraint_;
    DeviceBuffer<tensor> uR_;
    DeviceBuffer<label> bCell_;
    DeviceBuffer<scalar> res_;
    double solveRes_[4] = {0, 0, 0, 0};
    double solveResL1_[4] = {0, 0, 0, 0};
    double identityMax_[4] = {0, 0, 0, 0};
    DeviceBuffer<scalar> xp_[4], idC_, idM_, idB_, onesB_;
    // sutherland (milestone H): mu and alphahe per cell and face, the laplacians rebuilt each step; wedge patches
    bool sutherland_ = false, wedge_ = false;
    DeviceBuffer<scalar> alphaC_, alphaB_, gms_, upper_;
    DeviceBuffer<tensor> cellT_;
    void rebuildLaplacian(const DeviceBuffer<scalar>& gammaC, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& negUpper,
                          DeviceBuffer<scalar>& gammaF);
    // after a solve: its remainder, unsigned remainder and identity error into slot k
    void account(int k, const DeviceLduView& A, const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& b,
                 const DeviceBuffer<scalar>& rho, const DeviceBuffer<scalar>& iC, const DeviceBuffer<scalar>& bC,
                 const DeviceBuffer<scalar>* tauB, scalar rDeltaT, const DeviceBuffer<scalar>* const* scale);
};

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
