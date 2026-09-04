// cf GPU offload: k-omega SST blending (F1/F2), cross-diffusion (CDkOmega), strain (S2), and the coefficient
// blend. Pointwise per-cell kernels mirroring kOmegaSSTBase.C (v2412). See the header for the formulas.
#include "device_komega_sst.cuh"
#include "device_kepsilon.cuh"   // DeviceWallData (shared wall geometry)
#include "nut_wall_function.cuh" // nutkWallFunctionValue / yPlusWall (shared wall-nut physics, "G0 IDENTICAL to eps WF")
#include "device_scalar_transport.cuh"  // deviceSolveScalarTransport scaffold + depsKernel/overrideKernel + nBlocks/TPB
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_ami.cuh"
#include "device_cyclic.cuh"
#include "device_interface.cuh"
#include "device_amg.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace brae {

namespace {
// TPB/nBlocks now come from device_scalar_transport.cuh (shared with device_kepsilon.cu / device_spalart.cu).
inline scalar yPlusLamHost(scalar kappa, scalar E) { scalar y = 11.0; for (int i = 0; i < 10; ++i) y = std::log(std::fmax(E*y, 1.0)) / kappa; return y; }


__global__
void s2Kernel(int nC, const scalar* __restrict__ gradU, scalar* __restrict__ S2)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];
    scalar s = 0.0;   // magSqr(symm(gradU)) = sum_ab symm_ab^2
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            const scalar sab = 0.5 * (t[a*3+b] + t[b*3+a]);
            s += sab * sab;
        }
    S2[c] = 2.0 * s;
}


__global__
void cdKernel(
    int nC,
    const scalar* __restrict__ gKx,
    const scalar* __restrict__ gKy,
    const scalar* __restrict__ gKz,
    const scalar* __restrict__ gOx,
    const scalar* __restrict__ gOy,
    const scalar* __restrict__ gOz,
    const scalar* __restrict__ om,
    scalar twoA2,
    scalar* __restrict__ CD)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar dot = gKx[c]*gOx[c] + gKy[c]*gOy[c] + gKz[c]*gOz[c];   // grad k . grad omega
    CD[c] = twoA2 * dot / om[c];
}


__global__
void f1Kernel(
    int nC,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ CD,
    scalar nu,
    scalar betaStar,
    scalar alphaOmega2,
    int lm,
    scalar* __restrict__ F1,
    const scalar* __restrict__ nuCell)   // compressible: per-cell nu = mu/rho (nullptr -> the scalar nu)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // The SST blending needs the LAMINAR kinematic viscosity. Incompressible cases have one value for the
    // whole field, so it arrives as a scalar; compressible mu is T-dependent, so nu = mu/rho varies per
    // cell and must arrive as a buffer. nullptr -> the scalar -> bit-identical incompressible behaviour.
    const scalar nuc = nuCell ? nuCell[c] : nu;

    const scalar kk = k[c], w = om[c], yy = y[c];
    const scalar CDplus = fmax(CD[c], (scalar)1.0e-10);
    const scalar a  = (1.0 / betaStar) * sqrt(kk) / (w * yy);
    const scalar b  = 500.0 * nuc / (yy * yy * w);
    const scalar cc = (4.0 * alphaOmega2) * kk / (CDplus * yy * yy);
    const scalar arg1 = fmin(fmin(fmax(a, b), cc), (scalar)10.0);
    const scalar a2 = arg1 * arg1;   // pow4(arg1) = (arg1^2)^2
    scalar f1 = tanh(a2 * a2);
    // kOmegaSSTLM override (OF kOmegaSSTLM.C:42-52): F1 = max(F1, F3), F3 = exp(-(Ry/120)^8), Ry = y*sqrt(k)/nu.
    // Forces F1->1 in the near-wall transitional band (Ry<120) -> keeps the model in k-omega inner mode + suppresses
    // the (F1-1)*CDkOmega cross-diffusion source there. Only for the LM path (base kOmegaSST has no F3).
    if (lm)
    {
        const scalar Ry = yy * sqrt(kk) / nuc;
        const scalar r = Ry / 120.0;
        const scalar r2 = r * r, r4 = r2 * r2;
        f1 = fmax(f1, exp(-(r4 * r4)));
    }
    F1[c] = f1;
}


__global__
void f2Kernel(
    int nC,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    scalar nu,
    scalar betaStar,
    scalar* __restrict__ F2,
    const scalar* __restrict__ nuCell)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar nuc = nuCell ? nuCell[c] : nu;

    const scalar kk = k[c], w = om[c], yy = y[c];
    const scalar a = (2.0 / betaStar) * sqrt(kk) / (w * yy);
    const scalar b = 500.0 * nuc / (yy * yy * w);
    const scalar arg2 = fmin(fmax(a, b), (scalar)100.0);
    F2[c] = tanh(arg2 * arg2);   // sqr(arg2)
}


__global__
void blendKernel(int nC, const scalar* __restrict__ F1, scalar psi1, scalar psi2, scalar* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) out[c] = F1[c] * (psi1 - psi2) + psi2;
}


// correctNut: nut = a1*k / max(a1*omega, b1*F2*sqrt(S2))
__global__
void nutSSTKernel(
    int nC,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ F2,
    const scalar* __restrict__ S2,
    scalar a1,
    scalar b1,
    scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar denom = fmax(a1 * om[c], b1 * F2[c] * sqrt(S2[c]));
    nut[c] = a1 * k[c] / denom;
}


// Pk = min(G, c1*betaStar*k*omega)
__global__
void pkKernel(
    int nC,
    const scalar* __restrict__ G,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    scalar c1betaStar,
    scalar* __restrict__ Pk)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    Pk[c] = fmin(G[c], c1betaStar * k[c] * om[c]);
}


// GbyNu = min(GbyNu0, (c1/a1)*betaStar*omega*max(a1*omega, b1*F2*sqrt(S2)))
__global__
void gbyNuLimitKernel(
    int nC,
    const scalar* __restrict__ GbyNu0,
    const scalar* __restrict__ om,
    const scalar* __restrict__ F2,
    const scalar* __restrict__ S2,
    scalar a1,
    scalar b1,
    scalar c1,
    scalar betaStar,
    scalar* __restrict__ GbyNu)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar denom = fmax(a1 * om[c], b1 * F2[c] * sqrt(S2[c]));
    GbyNu[c] = fmin(GbyNu0[c], (c1 / a1) * betaStar * om[c] * denom);
}


// D = (F1*(alpha1-alpha2)+alpha2)*nut + nu
__global__
void dEffKernel(
    int nC,
    const scalar* __restrict__ F1,
    const scalar* __restrict__ nut,
    scalar alpha1,
    scalar alpha2,
    scalar nu,
    scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    D[c] = (F1[c] * (alpha1 - alpha2) + alpha2) * nut[c] + nu;
}


// omega reaction: diag += V*(max(sp1,0) + beta*omega + max(sp2,0));
//                 source += V*gamma*GbyNu0lim - V*min(sp1,0)*omega - V*min(sp2,0)*omega
//   sp1 = (2/3)*gamma*divU   (SuSp),   beta*omega (Sp),   sp2 = (F1-1)*CDkOmega/omega   (SuSp)
__global__
void omegaReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ gamma,
    const scalar* __restrict__ beta,
    const scalar* __restrict__ GbyNu0,
    const scalar* __restrict__ F1,
    const scalar* __restrict__ CD,
    const scalar* __restrict__ om,
    const scalar* __restrict__ divU,
    scalar* __restrict__ diag,
    scalar* __restrict__ source,
    const scalar* __restrict__ rho)   // compressible: every term rho-weighted (nullptr -> incompressible)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // OF kOmegaSSTBase.C omega equation: the production, both SuSp terms and the beta*omega Sp are all
    // alpha*rho*(...). Same treatment as the k sink -- the flux already carries rho and the diffusivity is
    // scaled by the caller, so only the reaction pair is left. nullptr -> 1.0 -> bit-identical.
    const scalar rw = rho ? rho[c] : scalar(1);
    const scalar sp1 = (2.0/3.0) * gamma[c] * divU[c];
    const scalar sp2 = (F1[c] - 1.0) * CD[c] / om[c];
    diag[c]   += rw * V[c] * (fmax(sp1, 0.0) + beta[c]*om[c] + fmax(sp2, 0.0));
    source[c] += rw * (V[c] * gamma[c]*GbyNu0[c] - V[c]*fmin(sp1, 0.0)*om[c] - V[c]*fmin(sp2, 0.0)*om[c]);
}


// k reaction: diag += V*(betaStar*omega + max(sp,0)); source += V*Pk(G) - V*min(sp,0)*k;  sp=(2/3)divU;
//             Pk = min(G, c1*betaStar*k*omega).  (k-eps kReaction with eps/k->betaStar*omega, G->Pk(G).)
__global__
void kReactionSSTKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ G,
    const scalar* __restrict__ divU,
    scalar betaStar,
    scalar c1betaStar,
    const scalar* __restrict__ gammaIntEff,
    scalar* __restrict__ diag,
    scalar* __restrict__ source,
    const scalar* __restrict__ FDES,   // kOmegaSST-DDES: k-dissipation *= FDES (nullptr -> 1 -> plain RANS)
    const scalar* __restrict__ rho)    // compressible: every term is rho-weighted (nullptr -> 1 -> incompressible)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // OF's compressible k equation is the incompressible one with rho on every term:
    //   fvm::div(alphaRhoPhi,k) - fvm::laplacian(alpha*rho*DkEff,k) == alpha*rho*Pk - fvm::Sp(alpha*rho*epsilonByk,k)
    // The flux already carries rho (phase 2) and the diffusivity is scaled by the caller, so all that is
    // left here is the production/sink pair. nullptr -> 1.0 -> bit-identical incompressible behaviour.
    const scalar rw = rho ? rho[c] : scalar(1);
    const scalar sp = (2.0/3.0) * divU[c];
    // kOmegaSSTLM transition: Pk *= gammaIntEff, epsilonByk (= betaStar*omega) *= clamp(gammaIntEff, 0.1, 1).
    // gammaIntEff == nullptr (plain kOmegaSST) -> geff = 1 -> bit-identical (clamp(1,0.1,1)=1).
    const scalar geff = gammaIntEff ? gammaIntEff[c] : 1.0;
    // kOmegaSST-DDES: the DES limiter FDES>=1 enhances the k destruction (beta*k*omega -> beta*k*omega*FDES) so the
    // modelled length scale collapses to the LES scale in detached regions. FDES==nullptr (RANS) -> factor 1 -> unchanged.
    const scalar fdes = FDES ? FDES[c] : scalar(1);
    const scalar Pk = geff * fmin(G[c], c1betaStar * k[c] * om[c]);
    diag[c]   += rw * V[c] * (fdes * fmin(fmax(geff, 0.1), 1.0) * betaStar * om[c] + fmax(sp, 0.0));
    source[c] += rw * (V[c] * Pk - V[c] * fmin(sp, 0.0) * k[c]);
}

// kOmegaSST-DDES DES factor: FDES = max( (Lt/(CDES*Delta))*(1 - F2), 1 ), Lt = sqrt(k)/(betaStar*omega) (RANS length),
// Delta = cubeRootVol = V^(1/3), CDES = F1*CDES1 + (1-F1)*CDES2 (SST-blended), F2 = the DDES shielding (RANS in the
// boundary layer where F2->1 -> FDES=1; LES in free shear where F2->0). Matches OF kOmegaSSTDDES.
__global__
void kOmegaSSTDESfactorKernel(
    int nC, const scalar* __restrict__ k, const scalar* __restrict__ om, const scalar* __restrict__ V,
    const scalar* __restrict__ F1, const scalar* __restrict__ F2, scalar betaStar, scalar CDES1, scalar CDES2,
    const scalar* __restrict__ dOpt, scalar* __restrict__ FDES)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar delta = dOpt ? dOpt[c] : cbrt(V[c]);
    const scalar Lt = sqrt(fmax(k[c], scalar(0))) / fmax(betaStar * om[c], 1e-300);
    const scalar CDES = F1[c]*CDES1 + (scalar(1) - F1[c])*CDES2;
    FDES[c] = fmax((Lt / fmax(CDES*delta, 1e-300)) * (scalar(1) - F2[c]), scalar(1));
}

// kOmegaSST-IDDES DES factor (Gritskevich/Garbaruk/Schuetze/Menter 2012): the improved (WMLES) length scale replaces the
// k-dissipation scale. lRAS = sqrt(k)/(betaStar*omega); lLES = CDES*Delta, CDES = F1*CDES1 + (1-F1)*CDES2, Delta =
// min(max(Cw*y, Cw*hmax), hmax) (hwn omitted). Blending: rd_t/rd_l from nut/nu, fdt/fl/ft/fB/fe as in SA-IDDES (the SST
// rd denominator sqrt(0.5(S^2+Omega^2)) equals |gradU|). lIDDES = fdTilde*(1+fe)*lRAS + (1-fdTilde)*lLES, and the k
// destruction beta*k*omega is scaled by FDES = lRAS/lIDDES (== k^(3/2)/lIDDES). NOT clamped to 1: the fe elevated-stress
// branch makes lIDDES > lRAS -> FDES < 1 (less destruction), which is the intended IDDES behaviour.
__global__
void kOmegaSSTIDDESfactorKernel(
    int nC, const scalar* __restrict__ k, const scalar* __restrict__ om, const scalar* __restrict__ F1,
    const scalar* __restrict__ gradU, const scalar* __restrict__ nut, const scalar* __restrict__ y,
    const scalar* __restrict__ hmax, const scalar* __restrict__ hwn, scalar nu, KOmegaSSTCoeffs co, scalar* __restrict__ FDES)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar g2 = 0;
    for (int q = 0; q < 9; ++q) { const scalar gk = gradU[q*nC + c]; g2 += gk*gk; }
    const scalar magGradU = fmax(sqrt(g2), scalar(1e-300));           // |gradU| = sqrt(0.5(S^2+Omega^2))
    const scalar lRAS = sqrt(fmax(k[c], scalar(0))) / fmax(co.betaStar*om[c], scalar(1e-300));   // RANS length (Lt)
    const scalar CDES = F1[c]*co.CDES1 + (scalar(1) - F1[c])*co.CDES2;
    const scalar hm = fmax(hmax[c], scalar(1e-300));
    const scalar delta = fmin(fmax(fmax(co.Cw*y[c], co.Cw*hm), hwn[c]), hm);   // IDDES delta = min(max(max(Cw*y,Cw*hmax),hwn), hmax)
    const scalar lLES = CDES*delta;
    const scalar kd2 = fmax(co.kappa*co.kappa*y[c]*y[c], scalar(1e-300));
    const scalar rdt = fmin(nut[c] / (magGradU*kd2), scalar(10));     // turbulent rd (SST nut)
    const scalar rdl = fmin(nu     / (magGradU*kd2), scalar(10));     // laminar rd (nu)
    const scalar adt = co.Cdt1*rdt;   const scalar fdt = scalar(1) - tanh(adt*adt*adt);
    const scalar al  = co.Cl*co.Cl*rdl; const scalar al2 = al*al, al4 = al2*al2, al8 = al4*al4;
    const scalar fl  = tanh(al8*al2);
    const scalar at  = co.Ct*co.Ct*rdt; const scalar ft = tanh(at*at*at);
    const scalar fe2 = scalar(1) - fmax(ft, fl);
    const scalar alpha = scalar(0.25) - y[c]/hm;
    const scalar fB  = fmin(scalar(2)*exp(scalar(-9)*alpha*alpha), scalar(1));
    const scalar fdTilde = fmax(scalar(1) - fdt, fB);
    const scalar fe1 = (alpha >= scalar(0)) ? scalar(2)*exp(scalar(-11.09)*alpha*alpha)
                                            : scalar(2)*exp(scalar(-9.0)*alpha*alpha);
    const scalar fe  = fmax(fe1 - scalar(1), scalar(0)) * fe2;
    const scalar lIDDES = fmax(fdTilde*(scalar(1) + fe)*lRAS + (scalar(1) - fdTilde)*lLES, scalar(1e-300));
    FDES[c] = lRAS / lIDDES;                                          // beta*k*omega scaling (== k^(3/2)/lIDDES)
}


// omega wall function (BINOMIAL n=2 default): omega0 = sqrt(omegaVis^2 + omegaLog^2), scattered to wall cells
// with cornerWeight invNw. G0 IDENTICAL to the epsilon wall function. Clone of device_kepsilon wallFnKernel.
__global__
void wallOmegaG0Kernel(
    int nWC,
    const label* __restrict__ wcCell,
    const label* __restrict__ wcStart,
    const label* __restrict__ wcFace,
    const scalar* __restrict__ wfY,
    const scalar* __restrict__ wfDc,
    const scalar* __restrict__ wux,
    const scalar* __restrict__ wuy,
    const scalar* __restrict__ wuz,
    const scalar* __restrict__ invNw,
    const scalar* __restrict__ k,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar nu,
    scalar yplLam,
    scalar Cmu25,
    scalar kappa,
    scalar E,
    scalar atmZ0,
    bool   atmBoundNut,
    scalar beta1,
    int nutWall,
    scalar* __restrict__ omega0,
    scalar* __restrict__ G0,
    const scalar* __restrict__ nuFace)   // compressible: per-wall-face nu, null -> the scalar nu
{
    // One thread per wall CELL, fixed face order, single write -- the kEpsilon twin. See buildDeviceWallData.
    const int wc = blockIdx.x * blockDim.x + threadIdx.x;
    if (wc >= nWC) return;

    const int c = wcCell[wc];
    const scalar kc = k[c], iN = invNw[c];
    scalar g0 = 0.0, w0 = 0.0;
    for (label j = wcStart[wc]; j < wcStart[wc+1]; ++j)
    {
        const label wf = wcFace[j];
        const scalar y = wfY[wf], dc = wfDc[wf];
        // OF omegaWallFunction and the near-wall G0 both read turbulenceModel::nu(patchi). Under Sutherland
        // with a hot wall that is several times the freestream value, so the scalar fallback is only for the
        // constant-property incompressible case.
        const scalar nuw = nuFace ? nuFace[wf] : nu;
        g0 += wallProductionG0(c, wf, y, dc, kc, iN, wux, wuy, wuz, Ux, Uy, Uz, nuw,
                               yplLam, Cmu25, kappa, E, atmZ0, atmBoundNut, nutWall);
        const scalar omegaVis = 6.0 * nuw / (beta1 * y * y);
        const scalar omegaLog = sqrt(kc) / (Cmu25 * kappa * y);
        w0 += iN * sqrt(omegaVis*omegaVis + omegaLog*omegaLog);   // BINOMIAL n=2 (distinct omega wall value)
    }
    G0[c]     = g0;
    omega0[c] = w0;
}
} // namespace


void deviceS2(const DeviceBuffer<scalar>& gradU, int nC, DeviceBuffer<scalar>& S2)
{
    S2.resize(nC);
    s2Kernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), S2.data());
    cudaCheck(cudaGetLastError(), "S2");
}


void deviceCDkOmega(
    const DeviceBuffer<scalar>& gKx,
    const DeviceBuffer<scalar>& gKy,
    const DeviceBuffer<scalar>& gKz,
    const DeviceBuffer<scalar>& gOx,
    const DeviceBuffer<scalar>& gOy,
    const DeviceBuffer<scalar>& gOz,
    const DeviceBuffer<scalar>& omega,
    scalar alphaOmega2,
    DeviceBuffer<scalar>& CD)
{
    const int nC = static_cast<int>(omega.size());
    CD.resize(nC);
    cdKernel<<<nBlocks(nC), TPB>>>(nC, gKx.data(), gKy.data(), gKz.data(), gOx.data(), gOy.data(), gOz.data(),
                                   omega.data(), 2.0 * alphaOmega2, CD.data());
    cudaCheck(cudaGetLastError(), "CDkOmega");
}


void deviceF1(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& CD,
    scalar nu,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& F1,
    bool lm,
    const DeviceBuffer<scalar>* nuCell)   // compressible per-cell nu = mu/rho; nullptr -> the scalar nu
{
    const int nC = static_cast<int>(k.size());
    F1.resize(nC);
    f1Kernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), y.data(), CD.data(), nu, co.betaStar, co.alphaOmega2, lm ? 1 : 0, F1.data(),
                                   nuCell ? nuCell->data() : nullptr);
    cudaCheck(cudaGetLastError(), "F1");
}


void deviceF2(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& y,
    scalar nu,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& F2,
    const DeviceBuffer<scalar>* nuCell)   // compressible per-cell nu = mu/rho; nullptr -> the scalar nu
{
    const int nC = static_cast<int>(k.size());
    F2.resize(nC);
    f2Kernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), y.data(), nu, co.betaStar, F2.data(),
                                   nuCell ? nuCell->data() : nullptr);
    cudaCheck(cudaGetLastError(), "F2");
}


void deviceBlend(const DeviceBuffer<scalar>& F1, scalar psi1, scalar psi2, DeviceBuffer<scalar>& out)
{
    const int nC = static_cast<int>(F1.size());
    out.resize(nC);
    blendKernel<<<nBlocks(nC), TPB>>>(nC, F1.data(), psi1, psi2, out.data());
    cudaCheck(cudaGetLastError(), "blend");
}


void deviceNutSST(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& F2,
    const DeviceBuffer<scalar>& S2,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& nut)
{
    const int nC = static_cast<int>(k.size());
    nut.resize(nC);
    nutSSTKernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), F2.data(), S2.data(), co.a1, co.b1, nut.data());
    cudaCheck(cudaGetLastError(), "nutSST");
}


void devicePk(
    const DeviceBuffer<scalar>& G,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& Pk)
{
    const int nC = static_cast<int>(k.size());
    Pk.resize(nC);
    pkKernel<<<nBlocks(nC), TPB>>>(nC, G.data(), k.data(), omega.data(), co.c1 * co.betaStar, Pk.data());
    cudaCheck(cudaGetLastError(), "Pk");
}


void deviceGbyNuLimit(
    const DeviceBuffer<scalar>& GbyNu0,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& F2,
    const DeviceBuffer<scalar>& S2,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& GbyNu)
{
    const int nC = static_cast<int>(omega.size());
    GbyNu.resize(nC);
    gbyNuLimitKernel<<<nBlocks(nC), TPB>>>(nC, GbyNu0.data(), omega.data(), F2.data(), S2.data(),
                                           co.a1, co.b1, co.c1, co.betaStar, GbyNu.data());
    cudaCheck(cudaGetLastError(), "GbyNuLimit");
}


void deviceDEff(
    const DeviceBuffer<scalar>& F1,
    const DeviceBuffer<scalar>& nut,
    scalar alpha1,
    scalar alpha2,
    scalar nu,
    DeviceBuffer<scalar>& D)
{
    const int nC = static_cast<int>(F1.size());
    D.resize(nC);
    dEffKernel<<<nBlocks(nC), TPB>>>(nC, F1.data(), nut.data(), alpha1, alpha2, nu, D.data());
    cudaCheck(cudaGetLastError(), "DEff");
}


void deviceOmegaReaction(
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& gamma,
    const DeviceBuffer<scalar>& beta,
    const DeviceBuffer<scalar>& GbyNu0lim,
    const DeviceBuffer<scalar>& F1,
    const DeviceBuffer<scalar>& CD,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& divU,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source,
    const DeviceBuffer<scalar>* rho)   // compressible rho weighting; nullptr -> incompressible (unchanged)
{
    const int nC = static_cast<int>(V.size());
    omegaReactionKernel<<<nBlocks(nC), TPB>>>(nC, V.data(), gamma.data(), beta.data(), GbyNu0lim.data(), F1.data(),
                                              CD.data(), omega.data(), divU.data(), diag.data(), source.data(),
                                              rho ? rho->data() : nullptr);
    cudaCheck(cudaGetLastError(), "omegaReaction");
}


// nuWall[i] = nuBnd[wfBndIdx[i]] -- the same OF nu(patchi) the nut wall functions read, re-indexed into the
// wall-face ordering that DeviceWallData (and therefore omegaWallFunction and the near-wall G0) uses.
__global__
void gatherWallNuK(
    int n,
    const label* __restrict__ idx,
    const scalar* __restrict__ nuBnd,
    scalar* __restrict__ nuWall)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nuWall[i] = nuBnd[idx[i]];
}

void deviceGatherWallNu(
    const DeviceBuffer<label>& wfBndIdx,
    const DeviceBuffer<scalar>& nuBnd,
    DeviceBuffer<scalar>& nuWall)
{
    const int n = static_cast<int>(wfBndIdx.size());
    if (n == 0 || nuBnd.size() == 0) return;
    nuWall.resize(n);
    gatherWallNuK<<<nBlocks(n), TPB>>>(n, wfBndIdx.data(), nuBnd.data(), nuWall.data());
    cudaCheck(cudaGetLastError(), "gatherWallNu");
}

// out[i] = cell[faceCell[i]] -- plain adjacent-cell extrapolation for a boundary face.
__global__
void gatherCellToFaceK(
    int n,
    const label* __restrict__ fc,
    const scalar* __restrict__ cellVal,
    scalar* __restrict__ faceVal)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    faceVal[i] = cellVal[fc[i]];
}

// nut at boundary FACES for the SST, evaluated the way OF fills it rather than extrapolated from the cell.
//
// OF sets nut by field assignment (kOmegaSSTBase::correctNut):
//     nut_ = a1*k/max(a1*omega, b1*F23*sqrt(S2));   nut_.correctBoundaryConditions();
// so a 'calculated' patch carries that expression evaluated on the BOUNDARY k, omega, F23 and S2 --
// correctBoundaryConditions() leaves such a patch alone. F23 = F2 unless the F3 option is on.
//
//     F2   = tanh(arg2^2),  arg2 = max(2 sqrt(k)/(betaStar omega y), 500 nu/(y^2 omega))
//     S2   = 2 magSqr(symm(gradU)),  gradU at the face = gradC + n (x) (snGrad - n & gradC)
//
// The boundary gradient is the same expression device_divdevreff.cu's gradBKernel uses (OF gaussGrad::
// correctBoundaryConditions), reproduced here so the SST does not depend on the stress module's layout.
__global__
void sstNutBoundaryK(
    int nB,
    const label* __restrict__ fc,
    const scalar* __restrict__ kBnd,
    const scalar* __restrict__ omBnd,
    const scalar* __restrict__ yCell,   // CELL wall distance. NOT the boundary bndY_, which brae stores as
                                        // 0 on every non-wall face -- and non-wall faces are exactly the
                                        // ones this evaluates. OF's y()[patchi] on a calculated patch is
                                        // the adjacent cell's wallDist, which is what this indexes.
    const scalar* __restrict__ nuB,      // per-face nu (compressible); null -> the scalar nu
    scalar nu,
    const scalar* __restrict__ gradU,    // CELL gradient (9 x nC)
    int nC,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    const scalar* __restrict__ uxb,
    const scalar* __restrict__ uyb,
    const scalar* __restrict__ uzb,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ dc,
    const label*  __restrict__ calcMask, // 1 where the nut BC is 'calculated'
    scalar a1,
    scalar b1,
    scalar betaStar,
    const scalar* __restrict__ nutCell,
    scalar* __restrict__ nutBnd)
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= nB) return;
    const int c = fc[bi];
    if (!calcMask || !calcMask[bi]) return;          // wall/zeroGradient patches keep what the caller set

    const scalar kb = kBnd[bi], ob = omBnd[bi], y = yCell[c];
    const scalar nuf = nuB ? nuB[bi] : nu;
    if (!(ob > scalar(0)) || !(y > scalar(0))) { nutBnd[bi] = nutCell[c]; return; }

    // boundary gradU, then S2
    scalar gc[9];
    for (int q = 0; q < 9; ++q) gc[q] = gradU[q*nC + c];
    const scalar nv[3] = { nx[bi], ny[bi], nz[bi] };
    const scalar sn[3] = { (uxb[bi]-Ux[c])*dc[bi], (uyb[bi]-Uy[c])*dc[bi], (uzb[bi]-Uz[c])*dc[bi] };
    scalar ngc[3];
    for (int j = 0; j < 3; ++j)
        ngc[j] = nv[0]*gc[0*3+j] + nv[1]*gc[1*3+j] + nv[2]*gc[2*3+j];
    scalar gb[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            gb[i*3+j] = gc[i*3+j] + nv[i]*(sn[j] - ngc[j]);
    scalar ss = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            const scalar sab = 0.5*(gb[a*3+b] + gb[b*3+a]);
            ss += sab*sab;
        }
    const scalar S2 = 2.0*ss;

    const scalar arg2 = fmax(2.0*sqrt(fmax(kb, scalar(0)))/(betaStar*ob*y), 500.0*nuf/(y*y*ob));
    const scalar F2   = tanh(arg2*arg2);
    nutBnd[bi] = a1*kb / fmax(a1*ob, b1*F2*sqrt(fmax(S2, scalar(0))));
}

void deviceSSTNutBoundary(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& kBnd,
    const DeviceBuffer<scalar>& omBnd,
    const DeviceBuffer<scalar>& yCell,
    const DeviceBuffer<scalar>* nuB,
    scalar nu,
    const DeviceBuffer<scalar>& gradU,
    int nC,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<label>& calcMask,
    const KOmegaSSTCoeffs& co,
    const DeviceBuffer<scalar>& nutCell,
    DeviceBuffer<scalar>& nutBnd)
{
    const int nB = dbU.comp[0].n;
    if (nB == 0 || nutBnd.size() != static_cast<std::size_t>(nB)) return;
    DeviceBuffer<scalar> uxb, uyb, uzb;
    deviceBCValue(dbU.comp[0], Ux, uxb);
    deviceBCValue(dbU.comp[1], Uy, uyb);
    deviceBCValue(dbU.comp[2], Uz, uzb);
    sstNutBoundaryK<<<nBlocks(nB), TPB>>>(nB, dbU.comp[0].faceCell.data(), kBnd.data(), omBnd.data(),
                                          yCell.data(), nuB ? nuB->data() : nullptr, nu,
                                          gradU.data(), nC, dbU.nx.data(), dbU.ny.data(), dbU.nz.data(),
                                          uxb.data(), uyb.data(), uzb.data(), Ux.data(), Uy.data(), Uz.data(),
                                          dbU.comp[0].deltaCoeffs.data(), calcMask.data(),
                                          co.a1, co.b1, co.betaStar, nutCell.data(), nutBnd.data());
    cudaCheck(cudaGetLastError(), "sstNutBoundary");
}

void deviceWallOmegaG0(
    const DeviceWallData& w,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    scalar nu,
    DeviceBuffer<scalar>& omega0,
    DeviceBuffer<scalar>& G0,
    const KOmegaSSTCoeffs& co,
    int nutWall,
    scalar atmZ0,
    bool   atmBoundNut,
    const DeviceBuffer<scalar>* nuFace)   // compressible: nu = mu_b/rho_b per WALL face (OF nu(patchi))
{
    const int nC = static_cast<int>(k.size());
    omega0.resize(nC); G0.resize(nC);
    cudaCheck(cudaMemsetAsync(omega0.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "omega0 zero");
    cudaCheck(cudaMemsetAsync(G0.data(),     0, nC*sizeof(scalar), cudaStreamPerThread), "G0 zero");
    const scalar Cmu25 = std::pow(co.betaStar, 0.25), yplLam = yPlusLamHost(co.kappa, co.E);
    if (w.nWC > 0)
        wallOmegaG0Kernel<<<nBlocks(w.nWC), TPB>>>(w.nWC, w.wcCell.data(), w.wcStart.data(), w.wcFace.data(),
                                                   w.wfY.data(), w.wfDc.data(), w.wfUwx.data(),
                                                   w.wfUwy.data(), w.wfUwz.data(), w.invNw.data(), k.data(), Ux.data(),
                                                   Uy.data(), Uz.data(), nu, yplLam, Cmu25, co.kappa, co.E, atmZ0, atmBoundNut, co.beta1,
                                                   nutWall, omega0.data(), G0.data(),
                                                   (nuFace && nuFace->size()) ? nuFace->data() : nullptr);
    cudaCheck(cudaGetLastError(), "wallOmegaG0");
}


void deviceKReactionSST(
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& G,
    const DeviceBuffer<scalar>& divU,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source,
    const scalar* gammaIntEff,
    const DeviceBuffer<scalar>* FDES,   // kOmegaSST-DDES DES factor per cell; nullptr -> plain RANS (unchanged)
    const DeviceBuffer<scalar>* rho)    // compressible rho weighting; nullptr -> incompressible (unchanged)
{
    const int nC = static_cast<int>(V.size());
    kReactionSSTKernel<<<nBlocks(nC), TPB>>>(nC, V.data(), k.data(), omega.data(), G.data(), divU.data(),
                                             co.betaStar, co.c1 * co.betaStar, gammaIntEff, diag.data(), source.data(),
                                             FDES ? FDES->data() : nullptr,
                                             rho ? rho->data() : nullptr);
    cudaCheck(cudaGetLastError(), "kReactionSST");
}

// Exported kOmegaSST-DDES DES-factor wrapper (single-GPU + unit-test hook): FDES from k/omega, cubeRootVol(V), F1, F2.
void deviceKOmegaSSTDESfactor(int nC, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& V, const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& F2,
    const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& FDES, const DeviceBuffer<scalar>* lesDelta)
{
    FDES.resize(nC);
    kOmegaSSTDESfactorKernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), V.data(), F1.data(), F2.data(),
                                                   co.betaStar, co.CDES1, co.CDES2,
                                                   (lesDelta && lesDelta->size()) ? lesDelta->data() : nullptr,
                                                   FDES.data());
    cudaCheck(cudaGetLastError(), "kOmegaSSTDESfactor");
}

// kOmegaSST-IDDES factor FDES = lRAS/lIDDES (Gritskevich et al. 2012). Needs gradU (|gradU| for rd), the SST nut, the
// wall distance y and the maxDeltaxyz hmax; F1 blends CDES. (Unit-test/DES hook.)
void deviceKOmegaSSTIDDESfactor(int nC, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& hmax, const DeviceBuffer<scalar>& hwn, scalar nu,
    const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& FDES)
{
    FDES.resize(nC);
    kOmegaSSTIDDESfactorKernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), F1.data(), gradU.data(),
        nut.data(), y.data(), hmax.data(), hwn.data(), nu, co, FDES.data());
    cudaCheck(cudaGetLastError(), "kOmegaSSTIDDESfactor");
}


// ---- kOmegaSST + Langtry-Menter transition (moved from device_kepsilon.cu; uses the SST helpers above + the scaffold) ----

// S2 (nut) and the production GbyNu0. When grad(U) is `cellLimited Gauss linear <k>` (motorBike), that gradient is
// LIMITED, else on skewed cells the unlimited strain is huge and the SST nut limiter max(a1*omega, b1*F2*sqrt(S2))
// wrongly fires. Apply the per-component minmod limiter to the gradU TENSOR:
// limiter_j scales gradU[j],[3+j],[6+j] (= dU_j/dx_i, i=0..2). Reuses deviceCellLimitGrad (the momentum limiter).
void deviceCellLimitGradU(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& gradU,
    scalar kc,
    const DeviceCyclic* cyc,
    const DeviceAMI* ami)
{
    const int nC = dm.nCells;
    const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
    // The AMI-interpolated (and rotated) neighbour velocity, once for all three components.
    DeviceBuffer<scalar> amiNbr[3];
    if (ami && ami->n > 0)
    {
        if (ami->rotational) deviceAmiInterpolateVec(*ami, Ux, Uy, Uz, amiNbr[0], amiNbr[1], amiNbr[2]);
        else for (int j = 0; j < 3; ++j) deviceAmiInterpolate(*ami, *U[j], amiNbr[j]);
    }
    for (int j = 0; j < 3; ++j)
    {
        DeviceBuffer<scalar> gx(nC), gy(nC), gz(nC);
        cudaMemcpyAsync(gx.data(), gradU.data()+(std::size_t)j*nC,     nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gy.data(), gradU.data()+(std::size_t)(3+j)*nC, nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gz.data(), gradU.data()+(std::size_t)(6+j)*nC, nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        DeviceBuffer<scalar> ubv;
        deviceBCValue(dbU.comp[j], *U[j], ubv);
        // The coupled patches join the limiter's range, exactly as in the momentum predictor -- an
        // interface cell limited as if the interface were not there is the defect this whole path
        // turned on. See CellLimitInterface.
        CellLimitInterface ifs[2];
        int nIfs = 0;
        DeviceBuffer<scalar> cycNbr;
        if (cyc && cyc->n > 0)
        {
            deviceCyclicNbrValue(*cyc, *U[j], Ux, Uy, Uz, j, cycNbr);
            ifs[nIfs++] = { cyc->n, cyc->ownCell.data(), cycNbr.data(),
                            cyc->dOwnX.data(), cyc->dOwnY.data(), cyc->dOwnZ.data() };
        }
        if (ami && ami->n > 0)
            ifs[nIfs++] = { ami->n, ami->ownCell.data(), amiNbr[j].data(),
                            ami->dOwnX.data(), ami->dOwnY.data(), ami->dOwnZ.data() };
        deviceCellLimitGrad(dm, *U[j], ubv, gx, gy, gz, kc, ifs, nIfs);
        cudaMemcpyAsync(gradU.data()+(std::size_t)j*nC,     gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gradU.data()+(std::size_t)(3+j)*nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gradU.data()+(std::size_t)(6+j)*nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }
    cudaCheck(cudaGetLastError(), "cellLimitGradUTensor");
}




// Elementwise a/b. Used for nu = mu/rho (cells) and for phi/interpolate(rho) (faces), which are the two
// places the SST has to undo a rho weighting that the rest of the compressible solver applies.
__global__
void nuFromMuRhoK(
    int n,
    const scalar* __restrict__ mu,
    const scalar* __restrict__ rho,
    scalar* __restrict__ nuOut)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nuOut[i] = mu[i] / rho[i];
}

// Compressible diffusivity for the k/omega equations.
//
// OF wants alpha*rho*DEff(F1) where DEff = alphaBlend*nut + nu (kinematic). Rather than re-derive that,
// deviceDEff is called with nu = 0 to get alphaBlend*nut, then scaled by rho and offset by the laminar
// DYNAMIC viscosity: rho*alphaBlend*nut + mu = alphaBlend*mut + mu, which is the same expression. Doing
// it this way means the SST blending itself is untouched and cannot drift from the incompressible path.
static void scaleDEffCompressible(
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& muLam,
    DeviceBuffer<scalar>& D)
{
    const int nC = static_cast<int>(D.size());
    DeviceBuffer<scalar> t;
    deviceHadamard(t, rho, D);
    deviceCopy(D, t);
    deviceAxpy(1.0, muLam, D);
    (void)nC;
}

void deviceKOmegaSSTCorrect(
    const DeviceMesh& dm,
    const DeviceWallData& wall,
    const DeviceBoundary& dbOmega,
    const DeviceBoundary& dbK,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& k,
    DeviceBuffer<scalar>& omega,
    DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relaxOmega,
    scalar relaxK,
    scalar tol,
    bool bounded,
    bool boundedEps,   // div(phi,epsilon|omega) `bounded`; `bounded` above is div(phi,k)'s
    bool limitedK,
    bool limitedOmega,
    scalar twoBykK,
    scalar twoBykOmega,
    const KOmegaSSTCoeffs& co,
    scalar relTolKE,
    int keCheckEvery,
    bool linearUpwindK,
    bool linearUpwindOmega,
    bool nonOrth,
    scalar gradULimitK,
    bool gsK,
    bool gsEps,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const scalar* gammaIntEff,
    int nutWall,
    scalar atmZ0,
    bool atmBoundNut,
    const ScalarDdt& kDdt,
    const ScalarDdt& sDdt,
    bool des,
    bool iddes,
    const DeviceBuffer<scalar>* hmax,
    const DeviceBuffer<scalar>* hwn,
    const DeviceBuffer<scalar>* rho,     // compressible: rho-weight the reactions and the diffusivity
    const DeviceBuffer<scalar>* muLam,   // compressible: laminar DYNAMIC viscosity mu [Pa s]
    const DeviceBuffer<scalar>* nuWallFace,   // compressible: nu = mu_b/rho_b per WALL face (OF nu(patchi))
    const DeviceBuffer<scalar>* rhoBnd,    // compressible: rho at boundary faces, for the volumetric flux
    const DeviceBuffer<scalar>* nutBnd,    // nut at boundary FACES -> DkEff/DomegaEff(patchi), as OF's laplacian uses
    const DeviceBuffer<scalar>* muBnd,
    scalar gradScalarLimitK,
    const DeviceBuffer<label>*  fvoKMask,
    const DeviceBuffer<scalar>* fvoKVal,
    const DeviceBuffer<label>*  fvoEMask,
    const DeviceBuffer<scalar>* fvoEVal,     // compressible: mu at boundary faces (the +mu of rho*D+mu)
    const DeviceBuffer<scalar>* lesDelta,    // case `delta` (maxDeltaxyz); null = OF's cubeRootVol
    int nSweeps)                             // solvers/<field>/nSweeps; see the declaration
{
    const int nC = dm.nCells;
    // production (raw GbyNu0) + G = nut*GbyNu0, divU, S2 (shared gradU = OF tgradU = grad(U) scheme).
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);   // interface-aware grad(U)
    if (gradULimitK > 0.0) deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradU, gradULimitK, cyc, ami);   // grad(U) cellLimited (OF)
    DeviceBuffer<scalar> GbyNu0;
    deviceGByNuFromGradU(gradU, nC, GbyNu0);
    DeviceBuffer<scalar> G;
    deviceHadamard(G, nut, GbyNu0);
    // TWO divergences, because OF uses two different fluxes here and they only coincide at constant rho:
    //
    //   divPhi -- div of the flux the convection term carries (the MASS flux when compressible). This is
    //             what "bounded" subtracts: boundedConvectionScheme::fvmDiv is
    //             scheme.fvmDiv(faceFlux,vf) - fvm::Sp(fvc::surfaceIntegrate(faceFlux), vf).
    //   divU   -- div of the VOLUMETRIC flux, phi/interpolate(rho). This is the dilatation in the k and
    //             omega reactions: kOmegaSSTBase.C builds it from compressibleTurbulenceModel::phi(),
    //             which is phi_/fvc::interpolate(rho_) whenever phi_ is not already volumetric.
    //
    // Feeding the mass-flux divergence to the reactions makes the (2/3)divU term wrong by a factor of rho
    // wherever density varies -- invisible in an isothermal case, ~1% in k at a hot wall.
    DeviceBuffer<scalar> divPhi;
    deviceDiv(dm, phiInt, phiBnd, divPhi);
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divPhi);
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divPhi);

    DeviceBuffer<scalar> divU;
    if (rho && rhoBnd && rhoBnd->size())
    {
        DeviceBuffer<scalar> rhoF;
        deviceInterpolate(dm, *rho, rhoF);
        const int nIf = dm.nInternalFaces;
        DeviceBuffer<scalar> phiVolInt;
        phiVolInt.resize(nIf);
        nuFromMuRhoK<<<nBlocks(nIf), TPB>>>(nIf, phiInt.data(), rhoF.data(), phiVolInt.data());
        cudaCheck(cudaGetLastError(), "phiVolInt");
        const int nB = static_cast<int>(phiBnd.size());
        DeviceBuffer<scalar> phiVolBnd;
        phiVolBnd.resize(nB);
        nuFromMuRhoK<<<nBlocks(nB), TPB>>>(nB, phiBnd.data(), rhoBnd->data(), phiVolBnd.data());
        cudaCheck(cudaGetLastError(), "phiVolBnd");
        deviceDiv(dm, phiVolInt, phiVolBnd, divU);
        if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);
        if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);
    }
    else
    {
        deviceCopy(divU, divPhi);   // incompressible: the two fluxes are the same field, bit-identical
    }
    DeviceBuffer<scalar> S2;
    deviceS2(gradU, nC, S2);

    // omega wall function FIRST (OF updateCoeffs before CDkOmega): omega0/G0, override omega & G at wall cells, so
    // grad(omega)/CDkOmega/F1/F2 and the reaction all see the wall-corrected omega (matches kOmegaSSTBase::correct).
    DeviceBuffer<scalar> omega0, G0;
    deviceWallOmegaG0(wall, k, Ux, Uy, Uz, nu, omega0, G0, co, nutWall, atmZ0, atmBoundNut, nuWallFace);
    overrideKernel<<<nBlocks(nC), TPB>>>(nC, wall.isWallCell.data(), G0.data(), omega0.data(), G.data(), omega.data(),
                                          wall.wallW.size() ? wall.wallW.data() : nullptr);

    // CDkOmega from grad(k), grad(omega); F1, F2.
    DeviceBuffer<scalar> kbv;
    deviceBCValue(dbK, k, kbv);
    DeviceBuffer<scalar> kgx, kgy, kgz;
    deviceGaussGrad(dm, k, kbv, kgx, kgy, kgz);
    DeviceBuffer<scalar> obv;
    deviceBCValue(dbOmega, omega, obv);
    DeviceBuffer<scalar> ogx, ogy, ogz;
    deviceGaussGrad(dm, omega, obv, ogx, ogy, ogz);
    DeviceBuffer<scalar> CD;
    deviceCDkOmega(kgx, kgy, kgz, ogx, ogy, ogz, omega, co.alphaOmega2, CD);
    DeviceBuffer<scalar> F1;
    // F1/F2 blend on the KINEMATIC laminar viscosity, the diffusivity wants the DYNAMIC one. Only mu is
    // passed in; nu = mu/rho is derived here so the two can never drift apart at the call site.
    DeviceBuffer<scalar> nuCellBuf;
    const DeviceBuffer<scalar>* nuCell = nullptr;
    if (rho && muLam)
    {
        const int nCk = static_cast<int>(muLam->size());
        nuCellBuf.resize(nCk);
        nuFromMuRhoK<<<nBlocks(nCk), TPB>>>(nCk, muLam->data(), rho->data(), nuCellBuf.data());
        cudaCheck(cudaGetLastError(), "nuFromMuRho");
        nuCell = &nuCellBuf;
    }
    deviceF1(k, omega, y, CD, nu, co, F1, gammaIntEff != nullptr, nuCell);   // LM: F1=max(F1,F3) near-wall override
    DeviceBuffer<scalar> F2;
    deviceF2(k, omega, y, nu, co, F2, nuCell);
    // kOmegaSST-DDES: the DES factor FDES>=1 (from the RANS length scale sqrt(k)/(betaStar*omega) vs C_DES*cubeRootVol,
    // shielded by 1-F2) multiplies the k destruction below. des==false -> FDES stays empty -> plain kOmegaSST(-RANS).
    DeviceBuffer<scalar> FDES;
    if (des)
    {
        if (iddes && hmax && hwn)   // kOmegaSST-IDDES: the improved (WMLES) length scale (needs the SST nut + hmax + hwn + gradU + y)
            deviceKOmegaSSTIDDESfactor(nC, k, omega, F1, gradU, nut, y, *hmax, *hwn, nu, co, FDES);
        else                 // kOmegaSST-DDES: the F2-shielded cubeRootVol DES factor
            deviceKOmegaSSTDESfactor(nC, k, omega, dm.V, F1, F2, co, FDES, lesDelta);
    }

    // blends, limited production-by-nu, DomegaEff.
    DeviceBuffer<scalar> gamma;
    deviceBlend(F1, co.gamma1, co.gamma2, gamma);
    DeviceBuffer<scalar> beta;
    deviceBlend(F1, co.beta1,  co.beta2,  beta);
    DeviceBuffer<scalar> GbyNu0lim;
    deviceGbyNuLimit(GbyNu0, omega, F2, S2, co, GbyNu0lim);
    DeviceBuffer<scalar> DomegaEff;
    deviceDEff(F1, nut, co.alphaOmega1, co.alphaOmega2, rho ? scalar(0) : nu, DomegaEff);
    if (rho && muLam) scaleDEffCompressible(*rho, *muLam, DomegaEff);

    // omega equation (loose solve) with the near-wall setValues constraint (omega0)
    // OF's laplacian(alpha*rho*DomegaEff, omega) uses the PATCH diffusivity, DomegaEff(patchi) =
    // alphaOmega(F1_b)*nut_b + nu_b -- not the adjacent cell's. Same correction that took kEpsilon from
    // 5e-3 to 1e-13. F1 is taken at the adjacent cell: OF evaluates alphaOmega(F1) as a volScalarField
    // whose boundary is F1's boundary, and F1 is a calculated field, so its patch value is the assigned
    // expression; using the cell F1 is an approximation ONLY in the blend factor, not in nut_b itself.
    DeviceBuffer<scalar> DomB, DkB;
    if (nutBnd && nutBnd->size())
    {
        const int nB = static_cast<int>(nutBnd->size());
        // F1 extrapolated from the adjacent cell. NOT deviceBCValue(dbOmega, F1, ...) -- that applies
        // OMEGA's boundary conditions to F1, so a fixedValue omega inlet returns omega's refValue (400)
        // where F1 must lie in [0,1]. Measured: that made omega 100x worse (3.1e-5 -> 3.2e-3).
        DeviceBuffer<scalar> F1b;
        F1b.resize(nB);
        gatherCellToFaceK<<<nBlocks(nB), TPB>>>(nB, dbOmega.faceCell.data(), F1.data(), F1b.data());
        cudaCheck(cudaGetLastError(), "F1b");
        DomB.resize(nB); DkB.resize(nB);
        deviceDEff(F1b, *nutBnd, co.alphaOmega1, co.alphaOmega2, rho ? scalar(0) : nu, DomB);
        deviceDEff(F1b, *nutBnd, co.alphaK1,     co.alphaK2,     rho ? scalar(0) : nu, DkB);
        if (rho && rhoBnd && muBnd && muBnd->size())
        {
            scaleDEffCompressible(*rhoBnd, *muBnd, DomB);
            scaleDEffCompressible(*rhoBnd, *muBnd, DkB);
        }
    }
    deviceSolveScalarTransport(dm, dbOmega, omega, "omega", DomegaEff, phiInt, phiBnd, divPhi, boundedEps, limitedOmega, linearUpwindOmega, nonOrth, twoBykOmega,
                               relaxOmega, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceOmegaReaction(dm.V, gamma, beta, GbyNu0lim, F1, CD, omega, divU, diag, src, rho); },
                               &wall, &omega0, ami, cyc, sDdt, DomB.size() ? &DomB : nullptr, gradScalarLimitK,
                               true, fvoEMask, fvoEVal,
                               // limField/limGrad* null (a scalar builds its own limiter), precon null
                               // (Jacobi), then nSweeps -- the case's own smoothSolver sweep count.
                               /*limField*/nullptr, /*limGradX*/nullptr, /*limGradY*/nullptr,
                               /*limGradZ*/nullptr, /*precon*/nullptr, nSweeps);
    deviceBoundField(dm, omega, 1e-15);   // OF bound(omega_, omegaMin_)

    // k equation (loose solve)
    DeviceBuffer<scalar> DkEff;
    deviceDEff(F1, nut, co.alphaK1, co.alphaK2, rho ? scalar(0) : nu, DkEff);
    if (rho && muLam) scaleDEffCompressible(*rho, *muLam, DkEff);
    deviceSolveScalarTransport(dm, dbK, k, "k", DkEff, phiInt, phiBnd, divPhi, bounded, limitedK, linearUpwindK, nonOrth, twoBykK,
                               relaxK, tol, relTolKE, keCheckEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceKReactionSST(dm.V, k, omega, G, divU, co, diag, src, gammaIntEff, des ? &FDES : nullptr, rho); },
                               nullptr, nullptr, ami, cyc, kDdt, DkB.size() ? &DkB : nullptr, gradScalarLimitK,
                               true, fvoKMask, fvoKVal,
                               // limField/limGrad* null (a scalar builds its own limiter), precon null
                               // (Jacobi), then nSweeps -- the case's own smoothSolver sweep count.
                               /*limField*/nullptr, /*limGradX*/nullptr, /*limGradY*/nullptr,
                               /*limGradZ*/nullptr, /*precon*/nullptr, nSweeps);
    deviceBoundField(dm, k, 1e-15);   // OF bound(k_, kMin_)

    // correctNut (Bradshaw): nut = a1*k / max(a1*omega, b1*F2*sqrt(S2)).
    //
    // F2 is RECOMPUTED here from the just-solved, just-bounded k and omega. OF's correctNut is
    //     nut_ = a1_*k_/max(a1_*omega_, b1_*F23()*sqrt(S2));            (kOmegaSSTBase.C:123)
    // where F23() is a FRESH call, so F2 is evaluated with the post-solve fields; only S2 is the stale
    // pre-solve one, and OF passes that in deliberately as an argument.
    //
    // brae used the F2 computed above (line ~974) from the PRE-solve k/omega, pairing a stale F2 with
    // post-solve k and omega. F2 is a function of BOTH fields, so that is a one-iteration lag in the
    // k<->omega coupling: in the shear-limited branch nut = a1*k/(b1*F2*sqrt(S2)) is directly
    // proportional to 1/F2, and the resulting nut feeds G and DkEff/DomegaEff on the next iteration --
    // a lagged feedback loop between the two equations.
    //
    // Invisible at convergence (F2_stale == F2_fresh there), so neither the converged-field gates nor a
    // one-iteration-from-developed-state reproducer can see it. It only bites in the transient, which is
    // exactly the regime a second-order convection scheme sharpens: pitzDaily kOmegaSST with
    // linearUpwind on BOTH k and omega converged to 5e-5 and then went unstable, while either scalar
    // alone was fine -- the signature of a coupling term, not of either equation.
    deviceF2(k, omega, y, nu, co, F2, nuCell);
    deviceNutSST(k, omega, F2, S2, co, nut);
}


// kOmegaSSTLM (Langtry-Menter gamma-ReThetat transition)
// LM coeffs (OF defaults): ca1=2, ca2=0.06, ce1=1, ce2=50, cThetat=0.03, sigmaThetat=2; lambdaErr=1e-6, maxIter=10.
namespace {
// OpenFOAM's SMALL (doubleScalar.H: 1.0e-15), which is what kOmegaSSTLM's deltaU_ and deltaMin are built
// from -- NOT VSMALL (1e-300) and not an arbitrary tiny number. Us appears SQUARED in a denominator and
// delta sits under y in y/delta, so the value of the floor is part of the model where the flow stagnates.
constexpr scalar LM_SMALL = 1.0e-15;
// A hang guard, not the model: OpenFOAM iterates lambda without a cap.
constexpr int LM_LAMBDA_HARD_CAP = 1000;
}

// kOmegaSSTLM's coeffDict, OpenFOAM's defaults. maxLambdaIter is a WARNING threshold, not a cap -- see
// lmLaunchReThetatPrep.
namespace { struct LMCoeffs { scalar ca1=2.0, ca2=0.06, ce1=1.0, ce2=50.0, cThetat=0.03, sigmaThetat=2.0, lambdaErr=1e-6; int maxLambdaIter=10; }; }


// DReThetatEff = sigmaThetat*(nut + nu)  (NOT nut/sigma + nu, depsKernel can't express this).
__global__
void lmReDiffKernel(int nC, const scalar* __restrict__ nut, scalar sigma, scalar nu, scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) D[c] = sigma*(nut[c] + nu);
}


// diag += V*sp ; source += V*su  (apply a precomputed semi-implicit reaction).
__global__
void lmAddReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ sp,
    const scalar* __restrict__ su,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    diag[c] += V[c] * sp[c];
    source[c] += V[c] * su[c];
}


// per-cell strain helpers from the OF-convention gradU tensor (t[i*3+j] = dU_j/dx_i).
__device__ __forceinline__
void lmStrain(const scalar* t, scalar ux, scalar uy, scalar uz, scalar deltaU,
              scalar& S, scalar& Omega, scalar& Us, scalar& dUsds)
{
    scalar symSq = 0.0, skSq = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar sy = 0.5*(t[i*3+j]+t[j*3+i]), sk = 0.5*(t[i*3+j]-t[j*3+i]);
            symSq += sy*sy;
            skSq += sk*sk;
        }
    S = sqrt(2.0*symSq);
    Omega = sqrt(2.0*skSq);
    Us = fmax(sqrt(ux*ux+uy*uy+uz*uz), deltaU);
    const scalar Uv[3] = {ux, uy, uz};
    scalar num = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            num += Uv[i]*Uv[j]*t[i*3+j];
    dUsds = num/(Us*Us);
}


// Fthetat (OF kOmegaSSTLM::Fthetat), uses the lagged gammaInt; reused by both the ReThetat source and gammaSep.
__device__ __forceinline__
scalar lmFthetat(scalar S, scalar Omega, scalar Us, scalar nu, scalar y, scalar om,
                 scalar ReThetat, scalar gammaInt, scalar ce2)
{
    const scalar delta = fmax(375.0*Omega*nu*ReThetat*y/(Us*Us), LM_SMALL);
    const scalar ReOmega = y*y*om/nu;
    const scalar Fwake = exp(-(ReOmega/1e5)*(ReOmega/1e5));
    scalar ywd = y/delta; ywd *= ywd; ywd *= ywd;   // (y/delta)^4
    const scalar invCe2 = 1.0/ce2, b = (gammaInt - invCe2)/(1.0 - invCe2);
    return fmin(fmax(Fwake*exp(-ywd), 1.0 - b*b), 1.0);
}


// ReThetac(ReThetat) and Flength(ReThetat) empirical correlations (OF).
__device__ __forceinline__
scalar lmReThetac(scalar R)
{
    return (R <= 1870.0) ? R - 396.035e-2 + 120.656e-4*R - 868.230e-6*R*R + 696.506e-9*R*R*R - 174.105e-12*R*R*R*R
                         : R - 593.11 - 0.482*(R - 1870.0);
}


__device__ __forceinline__
scalar lmFlength(scalar R, scalar y, scalar om, scalar nu)
{
    scalar Fl;
    if (R < 400.0)       Fl = 398.189e-1 - 119.270e-4*R - 132.567e-6*R*R;
    else if (R < 596.0)  Fl = 263.404 - 123.939e-2*R + 194.548e-5*R*R - 101.695e-8*R*R*R;
    else if (R < 1200.0) Fl = 0.5 - 3e-4*(R - 596.0);
    else                 Fl = 0.3188;
    const scalar fs = y*y*om/(200.0*nu);
    const scalar Fsub = exp(-(fs*fs));
    return Fl*(1.0 - Fsub) + 40.0*Fsub;
}


// ReThetat reaction prep: ReThetat0 Newton loop + Pthetat; outputs sp=Pthetat, su=Pthetat*ReThetat0, and Fthetat.
__global__
void lmReThetatPrepKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ ReThetat,
    const scalar* __restrict__ gammaInt,
    scalar nu,
    scalar cThetat,
    scalar ce2,
    scalar deltaU,
    scalar lambdaErr,
    scalar* __restrict__ Fth,
    scalar* __restrict__ sp,
    scalar* __restrict__ su,
    int* __restrict__ maxIterOut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q*nC+c];
    scalar S, Omega, Us, dUsds;
    lmStrain(t, Ux[c], Uy[c], Uz[c], deltaU, S, Omega, Us, dUsds);
    const scalar kc = k[c], nuc = nu, yc = y[c], omc = om[c], ret = ReThetat[c], gi = gammaInt[c];
    const scalar Tu = fmax(100.0*sqrt((2.0/3.0)*kc)/Us, 0.027);
    scalar lambda = 0.0, thetat = 0.0;
    int iter = 0;
    for (;;)
    {
        const scalar lam0 = lambda;
        scalar Fl;
        if (Tu <= 1.3)
        {
            Fl = (dUsds <= 0.0) ? 1.0 - (-12.986*lambda - 123.66*lambda*lambda - 405.689*lambda*lambda*lambda)*exp(-pow(Tu/1.5, 1.5))
                                : 1.0 + 0.275*(1.0 - exp(-35.0*lambda))*exp(-Tu/0.5);
            thetat = (1173.51 - 589.428*Tu + 0.2196/(Tu*Tu))*Fl*nuc/Us;
        }
        else
        {
            Fl = (dUsds <= 0.0) ? 1.0 - (-12.986*lambda - 123.66*lambda*lambda - 405.689*lambda*lambda*lambda)*exp(-pow(Tu/1.5, 1.5))
                                : 1.0 + 0.275*(1.0 - exp(-35.0*lambda))*exp(-2.0*Tu);
            thetat = 331.50*pow(Tu - 0.5658, -0.671)*Fl*nuc/Us;
        }
        lambda = fmin(fmax(thetat*thetat/nuc*dUsds, -0.1), 0.1);
        ++iter;
        // OpenFOAM's loop is `do { ... } while (lambdaErr > lambdaErr_)` with NO cap -- maxLambdaIter is
        // only the threshold past which it WARNS. Stopping at maxLambdaIter would return a lambda that
        // has not met the case's own lambdaErr, which is a different correlation, silently. So the cap
        // here is a hang guard set far above any converging cell, and the count is reported so the host
        // can warn exactly where OpenFOAM warns and refuse where OpenFOAM would spin.
        if (fabs(lambda - lam0) <= lambdaErr || iter >= LM_LAMBDA_HARD_CAP) break;
    }
    if (maxIterOut) atomicMax(maxIterOut, iter);
    const scalar ReThetat0 = fmax(thetat*Us/nuc, 20.0);
    const scalar Fthetat = lmFthetat(S, Omega, Us, nuc, yc, omc, ret, gi, ce2);
    Fth[c] = Fthetat;
    const scalar Pthetat = (cThetat/(500.0*nuc/(Us*Us)))*(1.0 - Fthetat);   // cThetat/t, t=500*nu/Us^2
    sp[c] = Pthetat;
    su[c] = Pthetat*ReThetat0;
}


// gammaInt reaction prep: Pgamma, Egamma -> sp=ce1*Pgamma+ce2*Egamma, su=Pgamma+Egamma.
__global__
void lmGammaPrepKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ ReThetat,
    const scalar* __restrict__ gammaInt,
    scalar nu,
    scalar ca1,
    scalar ca2,
    scalar ce1,
    scalar ce2,
    scalar deltaU,
    scalar* __restrict__ sp,
    scalar* __restrict__ su)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q*nC+c];
    scalar S, Omega, Us, dUsds;
    lmStrain(t, Ux[c], Uy[c], Uz[c], deltaU, S, Omega, Us, dUsds);
    const scalar kc = k[c], omc = om[c], yc = y[c], gi = gammaInt[c];
    const scalar ReThetac = lmReThetac(ReThetat[c]);
    const scalar Rev = yc*yc*S/nu, RT = kc/(nu*omc);
    const scalar Fonset1 = Rev/(2.193*ReThetac);
    const scalar Fonset2 = fmin(fmax(Fonset1, Fonset1*Fonset1*Fonset1*Fonset1), 2.0);
    const scalar Fonset3 = fmax(1.0 - (RT/2.5)*(RT/2.5)*(RT/2.5), 0.0);
    const scalar Fonset = fmax(Fonset2 - Fonset3, 0.0);
    const scalar Fturb = exp(-(0.25*RT)*(0.25*RT)*(0.25*RT)*(0.25*RT));
    const scalar Pgamma = ca1*lmFlength(ReThetat[c], yc, omc, nu)*S*sqrt(gi*Fonset);
    const scalar Egamma = ca2*Omega*Fturb*gi;
    sp[c] = ce1*Pgamma + ce2*Egamma;
    su[c] = Pgamma + Egamma;
}


// gammaIntEff = max(gammaInt, gammaSep); gammaSep = min(2*max(Rev/(3.235*ReThetac)-1,0)*Freattach, 2)*Fthetat.
__global__
void lmGammaEffKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ ReThetat,
    const scalar* __restrict__ gammaInt,
    const scalar* __restrict__ Fth,
    scalar nu,
    scalar deltaU,
    scalar* __restrict__ gammaIntEff)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q*nC+c];
    scalar S, Omega, Us, dUsds;
    lmStrain(t, Ux[c], Uy[c], Uz[c], deltaU, S, Omega, Us, dUsds);
    const scalar ReThetac = lmReThetac(ReThetat[c]);
    const scalar Rev = y[c]*y[c]*S/nu, RT = k[c]/(nu*om[c]);
    const scalar Freattach = exp(-(RT/20.0)*(RT/20.0)*(RT/20.0)*(RT/20.0));
    const scalar gammaSep = fmin(2.0*fmax(Rev/(3.235*ReThetac) - 1.0, 0.0)*Freattach, 2.0)*Fth[c];
    gammaIntEff[c] = fmax(gammaInt[c], gammaSep);
}



// The lambda fixed point is the one part of this model that ITERATES per cell, and OpenFOAM neither caps
// it nor stops on it -- it warns past maxLambdaIter and carries on. Reproducing that on a GPU needs a
// hang guard, so the kernel caps at LM_LAMBDA_HARD_CAP and reports the worst count: past maxLambdaIter
// brae warns exactly where OpenFOAM warns, and at the hard cap it REFUSES, because a lambda that never
// met the case's own lambdaErr is a different correlation and silently returning it is the substitution
// this port exists to eliminate.
void lmLaunchReThetatPrep(
    int nC, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, scalar nu,
    const LMCoeffs& lm,
    DeviceBuffer<scalar>& Fth, DeviceBuffer<scalar>& spR, DeviceBuffer<scalar>& suR)
{
    Fth.resize(nC); spR.resize(nC); suR.resize(nC);
    DeviceBuffer<label> worst;
    worst.copyFrom(std::vector<label>(1, 0));
    lmReThetatPrepKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(),
        k.data(), omega.data(), y.data(), ReThetat.data(), gammaInt.data(), nu,
        lm.cThetat, lm.ce2, LM_SMALL, lm.lambdaErr,
        Fth.data(), spR.data(), suR.data(), worst.data());
    cudaCheck(cudaGetLastError(), "lmReThetatPrep");
    const label used = worst.host()[0];
    if (used >= LM_LAMBDA_HARD_CAP)
        throw std::runtime_error(
            "brae: kOmegaSSTLM's lambda iteration did not reach lambdaErr in "
            + std::to_string(LM_LAMBDA_HARD_CAP) + " iterations. OpenFOAM iterates this loop without a "
            "cap, so returning the partly-converged lambda would run a different ReThetat0 correlation "
            "than the case asks for. Refusing rather than substituting it.");
    if (used > lm.maxLambdaIter)
    {
        static int warned = 0;
        if (!warned++)
            std::fprintf(stderr, "brae WARNING: kOmegaSSTLM lambda iterations (%d) exceed maxLambdaIter "
                                 "(%d)\n", (int)used, lm.maxLambdaIter);
    }
}

void deviceKOmegaSSTLMCorrect(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary& dbReThetat,
    const DeviceBoundary& dbGammaInt,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,
    DeviceBuffer<scalar>& ReThetat,
    DeviceBuffer<scalar>& gammaInt,
    DeviceBuffer<scalar>& gammaIntEff,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relax,
    scalar tol,
    scalar relTolKE,
    int keCheckEvery,
    bool bounded,
    bool nonOrth,
    bool gsEps,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const ScalarDdt& reDdt,
    const ScalarDdt& giDdt,
    // div(phi,ReThetat) and div(phi,gammaInt). T3A asks for `bounded Gauss linearUpwind grad` on both,
    // which is a DIFFERENT matrix from upwind -- upwind's, plus a deferred gradient correction on the
    // source. Running upwind under the case's own scheme name was worth a factor of 12 on the _cpp
    // reference's end-to-end error, so it is threaded through rather than assumed.
    bool limitedLinear,
    bool linearUpwind)
{
    const int nC = dm.nCells;
    const LMCoeffs lm;
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);

    // ReThetat: DReThetatEff = sigmaThetat*(nut+nu); reaction = Pthetat*ReThetat0 - Sp(Pthetat). Fthetat stored for gammaSep.
    DeviceBuffer<scalar> Fth, spR, suR;
    lmLaunchReThetatPrep(nC, gradU, Ux, Uy, Uz, k, omega, y, ReThetat, gammaInt, nu, lm, Fth, spR, suR);
    DeviceBuffer<scalar> DRe(nC);
    lmReDiffKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), lm.sigmaThetat, nu, DRe.data());   // sigmaThetat*(nut+nu)
    deviceSolveScalarTransport(dm, dbReThetat, ReThetat, "ReThetat", DRe, phiInt, phiBnd, divU, bounded, limitedLinear, linearUpwind, nonOrth, 2.0,
                               relax, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ lmAddReactionKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), spR.data(), suR.data(), diag.data(), src.data()); },
                               nullptr, nullptr, ami, cyc, reDdt);
    deviceBoundField(dm, ReThetat, 0.0);

    // gammaInt: DgammaIntEff = nut+nu; reaction = Pgamma+Egamma - Sp(ce1*Pgamma+ce2*Egamma).
    DeviceBuffer<scalar> spG(nC), suG(nC);
    lmGammaPrepKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), nu, lm.ca1, lm.ca2, lm.ce1, lm.ce2, LM_SMALL, spG.data(), suG.data());
    cudaCheck(cudaGetLastError(), "lmGammaPrep");
    DeviceBuffer<scalar> DgI(nC);
    depsKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), 1.0, nu, DgI.data());   // nut/1 + nu
    deviceSolveScalarTransport(dm, dbGammaInt, gammaInt, "gammaInt", DgI, phiInt, phiBnd, divU, bounded, limitedLinear, linearUpwind, nonOrth, 2.0,
                               relax, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ lmAddReactionKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), spG.data(), suG.data(), diag.data(), src.data()); },
                               nullptr, nullptr, ami, cyc, giDdt);
    deviceBoundField(dm, gammaInt, 0.0);
    gammaIntEff.resize(nC);
    lmGammaEffKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), Fth.data(), nu, LM_SMALL, gammaIntEff.data());
    cudaCheck(cudaGetLastError(), "lmGammaEff");
}

// ---- Exported LM (kOmegaSSTLM transition) source-prep wrappers ----------------------------------------------
// The transition kernels + LMCoeffs are cell-local (only gradU needs the halo, which the DISTRIBUTED SST correct
// already builds). These thin wrappers expose them so parallelDeviceKOmegaSSTLMCorrect can reuse the exact same
// physics + coefficients, feeding the distributed scalar-transport core -- the realizableKE pattern, two equations.
void deviceLMReDiff(const DeviceBuffer<scalar>& nut, scalar nu, DeviceBuffer<scalar>& D)
{
    const int nC = static_cast<int>(nut.size()); const LMCoeffs lm; D.resize(nC);
    lmReDiffKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), lm.sigmaThetat, nu, D.data());   // sigmaThetat*(nut+nu)
    cudaCheck(cudaGetLastError(), "deviceLMReDiff");
}
void deviceLMReThetatPrep(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, scalar nu,
    DeviceBuffer<scalar>& Fth, DeviceBuffer<scalar>& spR, DeviceBuffer<scalar>& suR)
{
    const int nC = dm.nCells; const LMCoeffs lm;
    lmLaunchReThetatPrep(nC, gradU, Ux, Uy, Uz, k, omega, y, ReThetat, gammaInt, nu, lm, Fth, spR, suR);
}
void deviceLMGammaPrep(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, scalar nu,
    DeviceBuffer<scalar>& spG, DeviceBuffer<scalar>& suG)
{
    const int nC = dm.nCells; const LMCoeffs lm; spG.resize(nC); suG.resize(nC);
    lmGammaPrepKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), nu, lm.ca1, lm.ca2, lm.ce1, lm.ce2, LM_SMALL, spG.data(), suG.data());
    cudaCheck(cudaGetLastError(), "deviceLMGammaPrep");
}
void deviceLMGammaEff(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, const DeviceBuffer<scalar>& Fth,
    scalar nu, DeviceBuffer<scalar>& gammaIntEff)
{
    const int nC = dm.nCells; gammaIntEff.resize(nC);
    lmGammaEffKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), Fth.data(), nu, LM_SMALL, gammaIntEff.data());
    cudaCheck(cudaGetLastError(), "deviceLMGammaEff");
}
void deviceLMAddReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& sp, const DeviceBuffer<scalar>& su,
    DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source)
{
    lmAddReactionKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.V.data(), sp.data(), su.data(), diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "deviceLMAddReaction");
}

} // namespace brae
