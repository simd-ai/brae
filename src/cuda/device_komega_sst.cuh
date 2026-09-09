#pragma once
// k-omega SST blending + strain functions on the device (C3). Pointwise per-cell formulas, verbatim from
// kOmegaSSTBase.C (OpenFOAM v2412):
//   S2       = 2*magSqr(symm(gradU))
//   CDkOmega = (2*alphaOmega2)*(grad k . grad omega)/omega          (raw; max(.,1e-10) applied inside F1)
//   F1       = tanh(arg1^4),  arg1 = min(min(max((1/betaStar)*sqrt(k)/(omega*y), 500*nu/(y^2*omega)),
//                                          (4*alphaOmega2)*k/(CDplus*y^2)), 10)
//   F2       = tanh(arg2^2),  arg2 = min(max((2/betaStar)*sqrt(k)/(omega*y), 500*nu/(y^2*omega)), 100)
//   blend(F1,psi1,psi2) = F1*(psi1-psi2)+psi2
// Gradients are produced by deviceGaussGrad (k,omega) / deviceGbyNu-style gradU in the model driver; here the
// kernels are pointwise on flat fields. Validated vs OF on a frozen state (ctest gpu_komega_sst).
#include "device_buffer.cuh"
#include "device_boundary.cuh"   // DeviceVectorBoundary (SST nut boundary)
#include "komega_sst_coeffs.cuh"

namespace brae {

struct DeviceWallData;   // device_kepsilon.cuh (shared wall geometry: wfCell/wfY/wfDc/wfUw*/invNw/nWF)

// S2 = 2*magSqr(symm(gradU)). gradU is the 9*nC tensor [q*nC+c], q=0..8 (same layout as deviceGbyNu).
void deviceS2(const DeviceBuffer<scalar>& gradU, int nC, DeviceBuffer<scalar>& S2);

// CDkOmega = (2*alphaOmega2)*(gradK . gradOmega)/omega.
void deviceCDkOmega(const DeviceBuffer<scalar>& gKx, const DeviceBuffer<scalar>& gKy, const DeviceBuffer<scalar>& gKz,
                    const DeviceBuffer<scalar>& gOx, const DeviceBuffer<scalar>& gOy, const DeviceBuffer<scalar>& gOz,
                    const DeviceBuffer<scalar>& omega, scalar alphaOmega2, DeviceBuffer<scalar>& CD);

// F1 = tanh(arg1^4) (CD = raw CDkOmega; the max(.,1e-10) is applied inside).
void deviceF1(const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
              const DeviceBuffer<scalar>& CD, scalar nu, const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& F1, bool lm = false,
    const DeviceBuffer<scalar>* nuCell = nullptr);

// F2 = tanh(arg2^2).
void deviceF2(const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
              scalar nu, const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& F2,
    const DeviceBuffer<scalar>* nuCell = nullptr);

// out = F1*(psi1-psi2)+psi2 (gamma/beta/alphaK/alphaOmega blends).
void deviceBlend(const DeviceBuffer<scalar>& F1, scalar psi1, scalar psi2, DeviceBuffer<scalar>& out);

// C4 limiters (kOmegaSSTBase.C correctNut / Pk / GbyNu)
// correctNut (Bradshaw): nut = a1*k / max(a1*omega, b1*F23*sqrt(S2)), F23 = F2 (F3 off).
void deviceNutSST(const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& F2,
                  const DeviceBuffer<scalar>& S2, const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& nut);

// k production cap: Pk = min(G, c1*betaStar*k*omega).  (G = nut*GbyNu0, passed in.)
void devicePk(const DeviceBuffer<scalar>& G, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
              const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& Pk);

// omega-eqn production limiter: GbyNu = min(GbyNu0, (c1/a1)*betaStar*omega*max(a1*omega, b1*F2*sqrt(S2))).
void deviceGbyNuLimit(const DeviceBuffer<scalar>& GbyNu0, const DeviceBuffer<scalar>& omega,
                      const DeviceBuffer<scalar>& F2, const DeviceBuffer<scalar>& S2,
                      const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& GbyNu);

// C5 omega equation
// Effective diffusivity D = blend(F1, alpha1, alpha2)*nut + nu  (DomegaEff: alphaOmega1/2; DkEff: alphaK1/2).
void deviceDEff(const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& nut, scalar alpha1, scalar alpha2,
                scalar nu, DeviceBuffer<scalar>& D);

// omega reaction (adds to diag + source, the deviceSolveScalarTransport reaction). Mirrors the omega block of
// kOmegaSSTBase::correct() (incompressible): production gamma*GbyNu0lim (Su) - SuSp((2/3)gamma divU) -
// Sp(beta omega) - SuSp((F1-1)CDkOmega/omega).  SuSp(s): diag+=V*max(s,0), source-=V*min(s,0)*omega.
void deviceOmegaReaction(const DeviceBuffer<scalar>& V, const DeviceBuffer<scalar>& gamma, const DeviceBuffer<scalar>& beta,
                         const DeviceBuffer<scalar>& GbyNu0lim, const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& CD,
                         const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& divU,
                         DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source,
    const DeviceBuffer<scalar>* rho = nullptr);

// omega wall function (BINOMIAL n=2): omega0 = sqrt(omegaVis^2 + omegaLog^2) scattered to wall cells (cornerWeight
// invNw), G0 identical to the epsilon wall function. Zeroes omega0/G0 then scatters. (Validated end-to-end at C7.)
void deviceWallOmegaG0(const DeviceWallData& w, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& Ux,
                       const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz, scalar nu,
                       DeviceBuffer<scalar>& omega0, DeviceBuffer<scalar>& G0, const KOmegaSSTCoeffs& co, int nutWall = 0,
                       scalar atmZ0 = 0.0, bool atmBoundNut = true,   // z0>0 -> atmNutkWallFunction (rough) for the G0 wall nut
                       const DeviceBuffer<scalar>* nuFace = nullptr,   // compressible: nu = mu_b/rho_b per WALL face
                       // The STORED wall nut in WALL-face order, as omegaWallFunction reads it
                       // (omegaWallFunctionFvPatchScalarField.C:199-200). Null keeps the
                       // recomputed nutkWallFunction -- the kEpsilon twin has the same argument.
                       const DeviceBuffer<scalar>* nutwStored = nullptr);

// nut at boundary faces for the SST, evaluated as OF's field assignment fills it (see the .cu).
void deviceSSTNutBoundary(const DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& kBnd,
                          const DeviceBuffer<scalar>& omBnd, const DeviceBuffer<scalar>& yCell,
                          const DeviceBuffer<scalar>* nuB, scalar nu, const DeviceBuffer<scalar>& gradU, int nC,
                          const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                          const DeviceBuffer<label>& calcMask, const KOmegaSSTCoeffs& co,
                          const DeviceBuffer<scalar>& nutCell, DeviceBuffer<scalar>& nutBnd);

// C6 k equation
// k reaction (adds to diag + source). Mirrors the k block of kOmegaSSTBase::correct() (incompressible):
// production Pk(G)=min(G, c1*betaStar*k*omega) (Su) - SuSp((2/3)divU) - Sp(betaStar*omega) [= epsilonByk]. This is
// the k-eps kReaction with eps/k -> betaStar*omega and G -> Pk(G).
// gammaIntEff (optional, kOmegaSSTLM transition): scales Pk by gammaIntEff and epsilonByk by clamp(gammaIntEff,0.1,1)
// per cell. nullptr (plain kOmegaSST) -> geff=1 -> bit-identical.
void deviceKReactionSST(const DeviceBuffer<scalar>& V, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
                        const DeviceBuffer<scalar>& G, const DeviceBuffer<scalar>& divU,
                        const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source,
                        const scalar* gammaIntEff = nullptr,
                        const DeviceBuffer<scalar>* FDES = nullptr,
    const DeviceBuffer<scalar>* rho = nullptr);   // kOmegaSST-DDES: k-dissipation *= FDES (nullptr=RANS)

// kOmegaSST-DDES DES factor FDES = max((sqrt(k)/(betaStar*omega))/(CDES*cubeRootVol(V))*(1-F2), 1), CDES=F1-blended.
// lesDelta: the case's LES filter width (`delta maxDeltaxyz`); nullptr keeps OF's cubeRootVol.
void deviceKOmegaSSTDESfactor(int nC, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& V, const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& F2,
    const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& FDES,
                              const DeviceBuffer<scalar>* lesDelta = nullptr);
// kOmegaSST-IDDES factor FDES = lRAS/lIDDES (Gritskevich et al. 2012): the improved (WMLES) length scale. (Unit-test/DES hook.)
void deviceKOmegaSSTIDDESfactor(int nC, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& hmax, const DeviceBuffer<scalar>& hwn, scalar nu,
    const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& FDES);

} // namespace brae
