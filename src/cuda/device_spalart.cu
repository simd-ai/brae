// SpalartAllmaras (one-equation) turbulence model: the nuTilda transport + its coefficients (Stilda/fw/fv1/fv2),
// transcribed from SpalartAllmarasBase::correct(). Uses the shared scalar-transport scaffold (device_scalar_transport.cuh)
// and the public helpers deviceGradU / deviceBoundaryNutSpalding. Verbatim split of device_kepsilon.cu -- no logic change.
#include "device_kepsilon.cuh"          // deviceSpalartAllmaras* decls + deviceGradU / deviceBoundaryNutSpalding / ScalarSolveEntry
#include "device_scalar_transport.cuh"  // deviceSolveScalarTransport scaffold + nBlocks/TPB/turbStore
#include "spalart_coeffs.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_ami.cuh"
#include "device_cyclic.cuh"
#include "device_interface.cuh"
#include "device_amg.cuh"
#include "nut_wall_function.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace brae {

// Spalart-Allmaras (one-equation)
// nuTilda transport, transcribed from SpalartAllmarasBase::correct() (incompressible, steady, ft2=off):
//   div(phi,nuTilda) - laplacian((nuTilda+nu)/sigma, nuTilda) + Sp(Cw1*fw*nuTilda/y^2)
//     == Cb1*Stilda*nuTilda + (Cb2/sigma)*|grad nuTilda|^2 ;   nut = nuTilda*fv1.
namespace {
// SA-DES/DDES/IDDES low-Re correction Psi (Spalart, Deck, Shur, Squires, Strelets, Travin 2006): multiplies the LES
// length scale, lLES = Psi*CDES*delta, so the RANS and LES branches stay consistent where the SA damping functions
// (fv1, fv2) reduce the eddy viscosity at low Reynolds number. With ft2=off (brae's SA):
//   Psi^2 = min(100, (1 - Cb1/(Cw1*kappa^2*fwStar)*fv2) / fv1),  fv1 = chi^3/(chi^3+Cv1^3),  fv2 = 1 - chi/(1+chi*fv1).
// Psi -> 1 at high chi (log layer / free shear); Psi grows (clamped at 10) in the viscous near-wall region, keeping the
// model RANS there. The numerator is >= 1-Cb1/(Cw1*kappa^2*fwStar) > 0 (fv2 <= 1), so the sqrt argument is positive.
__device__ inline scalar saPsi(scalar chi, const SpalartAllmarasCoeffs& co)
{
    const scalar chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const scalar fv1 = chi3 / (chi3 + Cv13);
    const scalar fv2 = scalar(1) - chi / (scalar(1) + chi*fv1);
    const scalar K = co.Cb1 / (co.Cw1() * co.kappa*co.kappa * co.fwStar);
    const scalar psi2 = fmin(scalar(100), (scalar(1) - K*fv2) / fmax(fv1, scalar(1e-300)));
    return sqrt(fmax(psi2, scalar(0)));
}

// Stilda = max(Omega + fv2*nuTilda/(kappa*y)^2, Cs*Omega); Omega = sqrt(2)*mag(skew(gradU)) (vorticity magnitude).
__global__
void saStildaKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ nt,
    const scalar* __restrict__ y,
    scalar nu,
    SpalartAllmarasCoeffs co,
    scalar* __restrict__ Stilda)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar chi = nt[c] / nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const scalar fv1 = chi3 / (chi3 + Cv13);
    const scalar fv2 = 1.0 - chi / (1.0 + chi*fv1);
    const scalar a = gradU[1*nC+c]-gradU[3*nC+c], b = gradU[2*nC+c]-gradU[6*nC+c], d = gradU[5*nC+c]-gradU[7*nC+c];
    const scalar Omega = sqrt(a*a + b*b + d*d);   // sqrt(2)*mag(skew(gradU))
    const scalar kd2 = fmax(co.kappa*y[c]*co.kappa*y[c], 1e-300);
    Stilda[c] = fmax(Omega + fv2*nt[c]/kd2, co.Cs*Omega);
}


// fw = g*((1+Cw3^6)/(g^6+Cw3^6))^(1/6),  g = r + Cw2*(r^6 - r),  r = min(nuTilda/(Stilda*(kappa*y)^2), 10).
__global__
void saFwKernel(
    int nC,
    const scalar* __restrict__ nt,
    const scalar* __restrict__ Stilda,
    const scalar* __restrict__ y,
    SpalartAllmarasCoeffs co,
    scalar* __restrict__ fw)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar kd2 = fmax(co.kappa*y[c]*co.kappa*y[c], 1e-300);
    scalar r = fmin(nt[c] / (fmax(Stilda[c], 1e-300) * kd2), 10.0);
    const scalar r6 = r*r*r*r*r*r;
    const scalar g = r + co.Cw2*(r6 - r), g6 = g*g*g*g*g*g, Cw36 = pow(co.Cw3, 6.0);
    fw[c] = g * pow((1.0 + Cw36) / (g6 + Cw36), 1.0/6.0);
}


__global__
void saReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ nt,
    const scalar* __restrict__ Stilda,
    const scalar* __restrict__ fw,
    const scalar* __restrict__ y,
    const scalar* __restrict__ gradNt2,
    SpalartAllmarasCoeffs co,
    scalar* __restrict__ diag,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar y2 = fmax(y[c]*y[c], 1e-300);
    diag[c] += V[c] * co.Cw1() * fw[c] * nt[c] / y2;                                          // destruction Sp
    src[c]  += V[c] * (co.Cb1 * Stilda[c] * nt[c] + (co.Cb2/co.sigmaNut) * gradNt2[c]);       // production + Cb2 grad^2
}


__global__
void saDEffKernel(int nC, const scalar* __restrict__ nt, scalar nu, scalar sigma, scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) D[c] = (nt[c] + nu) / sigma;
}


__global__
void saNutKernel(int nC, const scalar* __restrict__ nt, scalar nu, scalar Cv1, scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar chi = nt[c]/nu, chi3 = chi*chi*chi, Cv13 = Cv1*Cv1*Cv1;
    nut[c] = nt[c] * (chi3 / (chi3 + Cv13));   // nut = nuTilda*fv1
}


__global__
void saMagSqrKernel(
    int nC,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    scalar* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) out[c] = gx[c]*gx[c]+gy[c]*gy[c]+gz[c]*gz[c];
}

// SA-DDES length scale dTilda that REPLACES the wall distance y in the SA destruction (and Stilda/fw): the delayed
// detached-eddy limiter dTilda = y - fd*max(0, y - CDES*Delta), Delta = cubeRootVol = V^(1/3). The DDES shielding
// fd = 1 - tanh((8 rd)^3), rd = (nut+nu)/(|gradU| kappa^2 y^2) clamped to 10, keeps RANS in the boundary layer (fd->0 ->
// dTilda=y) and switches to LES in detached regions (fd->1 -> dTilda=CDES*Delta). |gradU| = sqrt(sum_ij (dU_i/dx_j)^2)
// (full 9-cmpt tensor magnitude); nut recomputed from nuTilda via fv1. lLES = Psi*CDES*Delta with the low-Re Psi (ft2=off).
// Matches OF SpalartAllmarasDDES. dTilda == y wherever fd==0, so a RANS (des=false) run is untouched.
__global__
void saDdesDTildaKernel(
    int nC, const scalar* __restrict__ y, const scalar* __restrict__ V, const scalar* __restrict__ gradU,
    const scalar* __restrict__ nt, scalar nu, SpalartAllmarasCoeffs co,
    const scalar* __restrict__ dOpt, const scalar* __restrict__ fdIn, scalar* __restrict__ dTilda)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar delta = dOpt ? dOpt[c] : cbrt(V[c]);                  // LES filter width: the case's `delta`
    scalar g2 = 0;
    for (int k = 0; k < 9; ++k) { const scalar gk = gradU[k*nC + c]; g2 += gk*gk; }   // |gradU|^2 (all 9 components)
    const scalar chi = nt[c]/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const scalar nutc = nt[c] * (chi3 / (chi3 + Cv13));                 // nut = nuTilda*fv1
    const scalar kd2 = fmax(co.kappa*co.kappa*y[c]*y[c], 1e-300);
    const scalar rd = fmin((nutc + nu) / (fmax(sqrt(g2), 1e-300) * kd2), scalar(10));
    // fd comes in ready-made on the ZDES2020 path (saZdesFdKernel); otherwise it is the standard
    // shielding, with OF's dictionary-exposed Cd1/Cd2 rather than the literals 8 and 3.
    const scalar t8 = co.Cd1 * rd;
    const scalar fd = fdIn ? fdIn[c] : (scalar(1) - tanh(pow(t8, co.Cd2)));
    const scalar lLES = saPsi(chi, co) * co.CDES * delta;              // low-Re-corrected LES length scale
    dTilda[c] = y[c] - fd * fmax(scalar(0), y[c] - lLES);
}

// |curl U| per CELL, from the packed cell gradient. OF fvc::curl = 2*(*skew(grad U)) with the Hodge dual
// *T = (T.yz, -T.xz, T.xy), which reduces to the textbook curl once gradU is read OF's way (T_ij = dU_j/dx_i):
//     curl = (dUz/dy - dUy/dz,  dUx/dz - dUz/dx,  dUy/dx - dUx/dy)
__global__
void saMagCurlKernel(int nC, const scalar* __restrict__ gradU, scalar* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar g1 = gradU[1*nC + c], g2 = gradU[2*nC + c], g3 = gradU[3*nC + c];
    const scalar g5 = gradU[5*nC + c], g6 = gradU[6*nC + c], g7 = gradU[7*nC + c];
    const scalar cx = g5 - g7, cy = g6 - g2, cz = g1 - g3;
    out[c] = sqrt(cx*cx + cy*cy + cz*cz);
}

// ...and per BOUNDARY FACE. |curl U| is an expression in grad(U), so its boundary value is whatever
// grad(U)'s boundary is -- and OF's gaussGrad::correctBoundaryConditions does NOT leave that as the
// extrapolated cell gradient: it replaces the wall-normal part with the patch snGrad,
//     gradU_b = gradU_c + n (x) (snGrad - n & gradU_c),   snGrad = (U_b - U_c)*deltaCoeffs
// which is the entire difference between a no-slip wall having a shear gradient and having none. Same
// expression as device_divdevreff.cu's gradBKernel and the SST's sstNutBoundaryK.
__global__
void saMagCurlBndKernel(
    int nB, int nC,
    const label*  __restrict__ fc,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ nx, const scalar* __restrict__ ny, const scalar* __restrict__ nz,
    const scalar* __restrict__ uxb, const scalar* __restrict__ uyb, const scalar* __restrict__ uzb,
    const scalar* __restrict__ Ux, const scalar* __restrict__ Uy, const scalar* __restrict__ Uz,
    const scalar* __restrict__ dc,
    scalar* __restrict__ out)
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= nB) return;
    const int c = fc[bi];
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
    const scalar cx = gb[1*3+2] - gb[2*3+1];
    const scalar cy = gb[2*3+0] - gb[0*3+2];
    const scalar cz = gb[0*3+1] - gb[1*3+0];
    out[bi] = sqrt(cx*cx + cy*cy + cz*cz);
}

// ZDES2020 shielding (Deck & Renard 2020), OF SpalartAllmarasDDES::fd with `shielding ZDES2020`:
//
//     r          = min(nuEff/(max(|gradU|,SMALL) (kappa y)^2), 10)          [the standard DDES ratio]
//     fdStd      = 1 - tanh((Cd1 r)^Cd2)
//     GnuTilda   = Cd3 max(grad(nuTilda).n, 0) / (max(|gradU|,SMALL) kappa y)
//     fdGnuTilda = 1 - tanh((Cd1 GnuTilda)^Cd2)          [* the fP2 factor when usefP2]
//     GOmega     = -(grad(|curl U|).n) sqrt(nuTilda/max(|gradU|^3, SMALL))
//     alpha      = (7/6 Cd4 - GOmega)/(Cd4/6)
//     fR(GOmega) = pos(Cd4 - GOmega)
//                + 1/(1 + exp(min(-6 alpha/max(1 - alpha^2, SMALL), 50))) pos(4Cd4/3 - GOmega) pos(GOmega - Cd4)
//     fd         = fdStd (1 - (1 - fdGnuTilda) fR)
//
// n is the OUTWARD wall normal of the nearest wall face (OF wallDist::n(), see cellWallDist). The two
// new gradients are wall-normal derivatives: GnuTilda detects the eddy-viscosity ramp that says "this is
// still an attached boundary layer" and GOmega the vorticity ramp, and where either says so the second
// factor pulls fd back towards 0, i.e. back to RANS. pos() is OF's STRICT pos (x > 0), not pos0.
__global__
void saZdesFdKernel(
    int nC,
    const scalar* __restrict__ y,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ nt,
    scalar nu,
    const scalar* __restrict__ wnx, const scalar* __restrict__ wny, const scalar* __restrict__ wnz,
    const scalar* __restrict__ gnx, const scalar* __restrict__ gny, const scalar* __restrict__ gnz,
    const scalar* __restrict__ gox, const scalar* __restrict__ goy, const scalar* __restrict__ goz,
    SpalartAllmarasCoeffs co,
    scalar* __restrict__ fd)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    constexpr scalar SMALL = scalar(1e-15);

    scalar g2 = 0;
    for (int k = 0; k < 9; ++k) { const scalar gk = gradU[k*nC + c]; g2 += gk*gk; }
    const scalar magGradU = sqrt(g2);
    const scalar chi = nt[c]/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const scalar nutc = nt[c] * (chi3 / (chi3 + Cv13));
    const scalar kd2 = fmax(co.kappa*co.kappa*y[c]*y[c], scalar(1e-300));
    const scalar r = fmin((nutc + nu) / (fmax(magGradU, scalar(1e-300)) * kd2), scalar(10));
    const scalar fdStd = scalar(1) - tanh(pow(co.Cd1*r, co.Cd2));

    const scalar nvx = wnx[c], nvy = wny[c], nvz = wnz[c];
    const scalar dNt = fmax(gnx[c]*nvx + gny[c]*nvy + gnz[c]*nvz, scalar(0));
    const scalar GnuTilda = co.Cd3 * dNt
                          / fmax(fmax(magGradU, SMALL) * co.kappa * y[c], scalar(1e-300));
    scalar fdG = scalar(1) - tanh(pow(co.Cd1*GnuTilda, co.Cd2));
    if (co.usefP2)
        fdG *= (scalar(1) - tanh(pow(co.Cd1*co.betaZDES*r, co.Cd2))) / fmax(fdStd, SMALL);

    const scalar mg3 = magGradU*magGradU*magGradU;
    const scalar GOmega = -(gox[c]*nvx + goy[c]*nvy + goz[c]*nvz)
                        * sqrt(fmax(nt[c], scalar(0)) / fmax(mg3, SMALL));
    const scalar alpha = (scalar(7)/scalar(6)*co.Cd4 - GOmega) / (co.Cd4/scalar(6));
    const scalar denom = fmax(scalar(1) - alpha*alpha, SMALL);
    const scalar fR = ((co.Cd4 - GOmega > scalar(0)) ? scalar(1) : scalar(0))
                    + scalar(1)/(scalar(1) + exp(fmin(scalar(-6)*alpha/denom, scalar(50))))
                      * ((scalar(4)*co.Cd4/scalar(3) - GOmega > scalar(0)) ? scalar(1) : scalar(0))
                      * ((GOmega - co.Cd4 > scalar(0)) ? scalar(1) : scalar(0));

    fd[c] = fdStd * (scalar(1) - (scalar(1) - fdG)*fR);
}

// SA-IDDES (SpalartAllmarasIDDES, Shur/Spalart/Strelets/Travin 2008): the IMPROVED delayed-DES length scale, adding
// wall-modelled-LES capability over DDES. dTilda = fdTilde*(1 + fe)*lRAS + (1 - fdTilde)*lLES, lRAS = y (wall dist),
// lLES = CDES*Delta with the IDDES delta Delta = min(max(Cw*y, Cw*hmax), hmax) (hmax = maxDeltaxyz; the wall-normal
// spacing hwn term is omitted -> hwn=0, a documented simplification). Blending: rd_t/rd_l from nut/nu, fdt/fl/ft the
// shielding + elevated-stress functions, fB/fe1/fe the WMLES branch. des==false leaves this path untaken (RANS exact).
__global__
void saIddesDTildaKernel(
    int nC, const scalar* __restrict__ y, const scalar* __restrict__ gradU,
    const scalar* __restrict__ nt, scalar nu, const scalar* __restrict__ hmax, const scalar* __restrict__ hwn,
    SpalartAllmarasCoeffs co, scalar* __restrict__ dTilda)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar g2 = 0;
    for (int k = 0; k < 9; ++k) { const scalar gk = gradU[k*nC + c]; g2 += gk*gk; }
    const scalar magGradU = fmax(sqrt(g2), scalar(1e-300));            // |gradU|
    const scalar chi = nt[c]/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const scalar nutc = nt[c] * (chi3 / (chi3 + Cv13));                // nut = nuTilda*fv1
    const scalar kd2 = fmax(co.kappa*co.kappa*y[c]*y[c], scalar(1e-300));
    const scalar rdt = fmin(nutc / (magGradU * kd2), scalar(10));      // turbulent rd (nut)
    const scalar rdl = fmin(nu   / (magGradU * kd2), scalar(10));      // laminar rd (nu)
    const scalar adt = co.Cdt1 * rdt;   const scalar fdt = scalar(1) - tanh(adt*adt*adt);   // fdt = 1 - tanh((Cdt1 rd_t)^3)
    const scalar al  = co.Cl*co.Cl*rdl; const scalar al2 = al*al, al4 = al2*al2, al8 = al4*al4;
    const scalar fl  = tanh(al8*al2);                                  // fl = tanh((Cl^2 rd_l)^10)
    const scalar at  = co.Ct*co.Ct*rdt; const scalar ft = tanh(at*at*at);                   // ft = tanh((Ct^2 rd_t)^3)
    const scalar fe2 = scalar(1) - fmax(ft, fl);
    const scalar hm  = fmax(hmax[c], scalar(1e-300));
    const scalar alpha = scalar(0.25) - y[c]/hm;
    const scalar fB  = fmin(scalar(2)*exp(scalar(-9)*alpha*alpha), scalar(1));
    const scalar fdTilde = fmax(scalar(1) - fdt, fB);
    const scalar fe1 = (alpha >= scalar(0)) ? scalar(2)*exp(scalar(-11.09)*alpha*alpha)
                                            : scalar(2)*exp(scalar(-9.0)*alpha*alpha);
    const scalar fe  = fmax(fe1 - scalar(1), scalar(0)) * fe2;
    const scalar delta = fmin(fmax(fmax(co.Cw*y[c], co.Cw*hm), hwn[c]), hm);   // IDDES delta = min(max(max(Cw*y,Cw*hmax),hwn), hmax)
    const scalar lLES  = saPsi(chi, co) * co.CDES * delta;            // low-Re-corrected LES length scale
    dTilda[c] = fmax(fdTilde*(scalar(1) + fe)*y[c] + (scalar(1) - fdTilde)*lLES, scalar(1e-300));
}
} // namespace (SA kernels)

void deviceSpalartAllmarasCorrect(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary& dbNuTilda,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& nuTilda,
    DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relax,
    scalar tol,
    bool bounded,
    bool limited,
    scalar twoByk,
    const SpalartAllmarasCoeffs& co,
    scalar relTol,
    int checkEvery,
    bool linearUpwind,
    bool nonOrth,
    bool gsK,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const ScalarDdt& ntDdt,
    bool des,
    bool iddes,
    const DeviceBuffer<scalar>* hmax,
    const DeviceBuffer<scalar>* hwn,
    const DeviceBuffer<scalar>* lesDelta,
    const DeviceBuffer<scalar>* wallN,
    scalar gradULimitK,   // OF `grad(U)` cellLimited coefficient; see the declaration
    int nSweeps)          // solvers/nuTilda/nSweeps; see the declaration
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);   // interface-aware grad(U) for vorticity/production
    // The case's own `grad(U)` scheme, which fvc::grad(U) resolves (SpalartAllmarasBase.C:461). Applied
    // to the TENSOR, as cellLimitedGrad does, so Omega, Stilda and the DES/IDDES length scales all see
    // the same gradient OpenFOAM built them from.
    if (gradULimitK > scalar(0)) deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradU, gradULimitK);
    // SA-DDES/IDDES: replace the wall distance y with the DES length scale dTilda in the SA length-scale terms (Stilda,
    // fw, destruction). des==false (RANS) -> dScale aliases y, so the model is byte-for-byte the standard SA. iddes uses
    // the improved (WMLES) length scale (needs hmax = maxDeltaxyz); otherwise the plain DDES cubeRootVol limiter.
    DeviceBuffer<scalar> dTilda;
    if (des)
    {
        dTilda.resize(static_cast<std::size_t>(nC));
        // ZDES2020 shielding needs two wall-normal derivatives that the standard fd does not:
        // grad(nuTilda).n and grad(|curl U|).n. Both are built here, from the SAME gradU the rest of the
        // model uses, and both need the wall normal wallN (3 x nC, packed) -- without it the case gets
        // the standard fd, which is the pre-ZDES behaviour rather than a wrong one.
        DeviceBuffer<scalar> zfd;
        if (co.zdes && !iddes && wallN && wallN->size() == static_cast<std::size_t>(3*nC))
        {
            DeviceBuffer<scalar> nbv0, gx0, gy0, gz0;
            deviceBCValue(dbNuTilda, nuTilda, nbv0);
            deviceGaussGrad(dm, nuTilda, nbv0, gx0, gy0, gz0);          // fvc::grad(nuTilda)

            DeviceBuffer<scalar> mc(static_cast<std::size_t>(nC)), mcb, ox, oy, oz;
            saMagCurlKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), mc.data());
            cudaCheck(cudaGetLastError(), "saMagCurl");
            const int nB = dbU.n;
            mcb.resize(static_cast<std::size_t>(nB));
            DeviceBuffer<scalar> ub[3];
            for (int k = 0; k < 3; ++k) deviceBCValue(dbU.comp[k], k == 0 ? Ux : (k == 1 ? Uy : Uz), ub[k]);
            saMagCurlBndKernel<<<nBlocks(nB), TPB>>>(nB, nC, dbU.comp[0].faceCell.data(), gradU.data(),
                                                     dbU.nx.data(), dbU.ny.data(), dbU.nz.data(),
                                                     ub[0].data(), ub[1].data(), ub[2].data(),
                                                     Ux.data(), Uy.data(), Uz.data(),
                                                     dbU.comp[0].deltaCoeffs.data(), mcb.data());
            cudaCheck(cudaGetLastError(), "saMagCurlBnd");
            deviceGaussGrad(dm, mc, mcb, ox, oy, oz);                   // fvc::grad(mag(curl(U)))

            zfd.resize(static_cast<std::size_t>(nC));
            const scalar* wn = wallN->data();
            saZdesFdKernel<<<nBlocks(nC), TPB>>>(nC, y.data(), gradU.data(), nuTilda.data(), nu,
                                                 wn, wn + nC, wn + 2*nC,
                                                 gx0.data(), gy0.data(), gz0.data(),
                                                 ox.data(), oy.data(), oz.data(), co, zfd.data());
            cudaCheck(cudaGetLastError(), "saZdesFd");
        }
        if (iddes && hmax && hwn)
            saIddesDTildaKernel<<<nBlocks(nC), TPB>>>(nC, y.data(), gradU.data(), nuTilda.data(), nu, hmax->data(), hwn->data(), co, dTilda.data());
        else
            saDdesDTildaKernel<<<nBlocks(nC), TPB>>>(nC, y.data(), dm.V.data(), gradU.data(), nuTilda.data(), nu, co,
                                                     (lesDelta && lesDelta->size()) ? lesDelta->data() : nullptr,
                                                     zfd.size() ? zfd.data() : nullptr, dTilda.data());
    }
    const DeviceBuffer<scalar>& dScale = des ? dTilda : y;
    DeviceBuffer<scalar> Stilda(static_cast<std::size_t>(nC));
    saStildaKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), nuTilda.data(), dScale.data(), nu, co, Stilda.data());
    DeviceBuffer<scalar> fw(static_cast<std::size_t>(nC));
    saFwKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), Stilda.data(), dScale.data(), co, fw.data());
    DeviceBuffer<scalar> nbv;
    deviceBCValue(dbNuTilda, nuTilda, nbv);   // |grad nuTilda|^2
    DeviceBuffer<scalar> gnx, gny, gnz;
    deviceGaussGrad(dm, nuTilda, nbv, gnx, gny, gnz);
    DeviceBuffer<scalar> gradNt2(static_cast<std::size_t>(nC));
    saMagSqrKernel<<<nBlocks(nC), TPB>>>(nC, gnx.data(), gny.data(), gnz.data(), gradNt2.data());
    if (const char* sapath = std::getenv("BRAE_DUMP_SA"))   // cell-level SA-term dump (vs OF reference) -- first call only
    {
        static bool saDumped = false;
        if (!saDumped)
        {
            saDumped = true;
            const auto hgU = gradU.host(); const auto hnt = nuTilda.host(); const auto hy = y.host();
            const auto hSt = Stilda.host(); const auto hfw = fw.host(); const auto hg2 = gradNt2.host();
            std::FILE* f = std::fopen(sapath, "w");
            std::fprintf(f, "cell,nuTilda,y,Omega,Stilda,fw,gradNt2,P,D,nut\n");
            for (int c = 0; c < nC; ++c)
            {
                const scalar a = hgU[1*nC+c]-hgU[3*nC+c], b = hgU[2*nC+c]-hgU[6*nC+c], d = hgU[5*nC+c]-hgU[7*nC+c];
                const scalar Om = std::sqrt(a*a + b*b + d*d);
                const scalar chi = hnt[c]/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1, fv1 = chi3/(chi3+Cv13);
                const scalar y2 = hy[c]*hy[c] > 1e-300 ? hy[c]*hy[c] : 1e-300;
                const scalar P = co.Cb1*hSt[c]*hnt[c] + (co.Cb2/co.sigmaNut)*hg2[c];
                const scalar D = co.Cw1()*hfw[c]*hnt[c]/y2;
                std::fprintf(f, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                             c, hnt[c], hy[c], Om, hSt[c], hfw[c], hg2[c], P, D, hnt[c]*fv1);
            }
            std::fclose(f);
            std::fprintf(stderr, "[BRAE_DUMP_SA] wrote %d cells to %s\n", nC, sapath);
        }
    }
    DeviceBuffer<scalar> D(static_cast<std::size_t>(nC));
    saDEffKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, co.sigmaNut, D.data());
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);   // for the bounded term
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);
    cudaCheck(cudaGetLastError(), "SA assemble");
    // OF's DnuTildaEff() is a volScalarField, (nuTilda + nu)/sigmaNut, so the LAPLACIAN's wall-face
    // coefficient uses the PATCH value of nuTilda (SpalartAllmarasBase.C:381-390, and
    // gaussLaplacianScheme.C uses gamma.boundaryField()). At an SA wall nuTilda is fixedValue 0, so OF's
    // wall diffusivity is exactly nu/sigmaNut.
    //
    // Without DBnd the fallback uses the adjacent CELL's D = (nuTilda_P + nu)/sigmaNut, i.e. too large by
    // (1 + nuTilda_P/nu) -- a median factor of ~133 on airFoil2D. That is a spurious implicit sink on every
    // wall-adjacent cell worth ~0.23/fw of the whole destruction term, which over-damps nuTilda exactly
    // where the measured error lives. It is SA-specific because the boundary laplacian weight is zero for a
    // zeroGradient patch, so the k/omega walls never saw it.
    DeviceBuffer<scalar> ntBnd, DB;
    deviceBCValue(dbNuTilda, nuTilda, ntBnd);
    if (ntBnd.size())
    {
        DB.resize(ntBnd.size());
        saDEffKernel<<<nBlocks(static_cast<int>(ntBnd.size())), TPB>>>(static_cast<int>(ntBnd.size()),
                                                                      ntBnd.data(), nu, co.sigmaNut, DB.data());
        cudaCheck(cudaGetLastError(), "SA DEff boundary");
    }
    deviceSolveScalarTransport(dm, dbNuTilda, nuTilda, "nuTilda", D, phiInt, phiBnd, divU, bounded, limited, linearUpwind, nonOrth, twoByk,
                               relax, tol, relTol, checkEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){
                                   saReactionKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), nuTilda.data(), Stilda.data(),
                                       fw.data(), dScale.data(), gradNt2.data(), co, diag.data(), src.data()); },
                               nullptr, nullptr, ami, cyc, ntDdt, DB.size() ? &DB : nullptr,
                               // gradLimitK 0, boundPositive true (nuTilda is positive-definite), no
                               // fvOptions set-values, no limiter override, Jacobi, then the case's own
                               // smoothSolver sweep count.
                               /*gradLimitK*/0.0, /*boundPositive*/true,
                               /*fvoSetMask*/nullptr, /*fvoSetVal*/nullptr,
                               /*limField*/nullptr, /*limGradX*/nullptr, /*limGradY*/nullptr,
                               /*limGradZ*/nullptr, /*precon*/nullptr, nSweeps);
    // deviceSolveScalarTransport already bounds to 1e-15 (~ bound(nuTilda, 0)). correctNut: nut = nuTilda*fv1(new).
    saNutKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, co.Cv1, nut.data());
    cudaCheck(cudaGetLastError(), "SA correctNut");
}


// Standalone SA correctNut (nut = nuTilda*fv1), used by the solver's startup validate() (OF
// eddyViscosity::validate()->correctNut()), so the FIRST momentum predictor sees a consistent nut.
// Boundary variant: nu VARIES per face on a compressible mesh (nu = mu/rho), so the scalar-nu form
// would be wrong there. nuFace == nullptr falls back to the uniform nu.
__global__
void saNutFaceKernel(int n, const scalar* __restrict__ nt, const scalar* __restrict__ nuFace,
                     scalar nu, scalar Cv1, scalar* __restrict__ nut)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar nuL = nuFace ? nuFace[i] : nu;
    const scalar chi = nt[i]/nuL, chi3 = chi*chi*chi, Cv13 = Cv1*Cv1*Cv1;
    nut[i] = nt[i] * (chi3 / (chi3 + Cv13));
}

void deviceNutSABoundary(const DeviceBuffer<scalar>& nuTildaB, const DeviceBuffer<scalar>* nuFace,
                         scalar nu, scalar Cv1, DeviceBuffer<scalar>& nutB)
{
    const int n = static_cast<int>(nuTildaB.size());
    nutB.resize(n);
    saNutFaceKernel<<<nBlocks(n), TPB>>>(n, nuTildaB.data(), nuFace ? nuFace->data() : nullptr,
                                         nu, Cv1, nutB.data());
    cudaCheck(cudaGetLastError(), "SA correctNut (boundary)");
}

void deviceNutSA(const DeviceBuffer<scalar>& nuTilda, scalar nu, scalar Cv1, DeviceBuffer<scalar>& nut)
{
    const int nC = static_cast<int>(nuTilda.size());
    nut.resize(nC);
    saNutKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, Cv1, nut.data());
    cudaCheck(cudaGetLastError(), "SA correctNut (validate)");
}

// Exported SA-DDES length-scale wrapper: dTilda from y, cubeRootVol(V), |gradU| and nuTilda (a unit-test hook + a future
// distributed-DES entry). Takes nC + V directly (no DeviceMesh) so it is callable without building a full mesh.
// Unit-test hook for the ZDES2020 shielding: fd from the eight fields it reads, no mesh required.
void deviceSAZdesFd(
    int nC, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& nuTilda, scalar nu,
    const DeviceBuffer<scalar>& wnx, const DeviceBuffer<scalar>& wny, const DeviceBuffer<scalar>& wnz,
    const DeviceBuffer<scalar>& gnx, const DeviceBuffer<scalar>& gny, const DeviceBuffer<scalar>& gnz,
    const DeviceBuffer<scalar>& gox, const DeviceBuffer<scalar>& goy, const DeviceBuffer<scalar>& goz,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& fd)
{
    fd.resize(static_cast<std::size_t>(nC));
    saZdesFdKernel<<<nBlocks(nC), TPB>>>(nC, y.data(), gradU.data(), nuTilda.data(), nu,
                                         wnx.data(), wny.data(), wnz.data(),
                                         gnx.data(), gny.data(), gnz.data(),
                                         gox.data(), goy.data(), goz.data(), co, fd.data());
    cudaCheck(cudaGetLastError(), "deviceSAZdesFd");
}


void deviceSADDESdTilda(int nC, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nuTilda, scalar nu,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& dTilda)
{
    dTilda.resize(nC);
    saDdesDTildaKernel<<<nBlocks(nC), TPB>>>(nC, y.data(), V.data(), gradU.data(), nuTilda.data(), nu, co, nullptr, nullptr, dTilda.data());
    cudaCheck(cudaGetLastError(), "deviceSADDESdTilda");
}

// Exported SA-IDDES length-scale wrapper: the improved (WMLES) dTilda from y, hmax (maxDeltaxyz), |gradU|, nuTilda
// (a unit-test hook + a future distributed entry). Uses hmax (not cubeRootVol) for the IDDES delta.
void deviceSAIDDESdTilda(int nC, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& hmax,
    const DeviceBuffer<scalar>& hwn, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nuTilda, scalar nu,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& dTilda)
{
    dTilda.resize(nC);
    saIddesDTildaKernel<<<nBlocks(nC), TPB>>>(nC, y.data(), gradU.data(), nuTilda.data(), nu, hmax.data(), hwn.data(), co, dTilda.data());
    cudaCheck(cudaGetLastError(), "deviceSAIDDESdTilda");
}

// ---- Exported SA (Spalart-Allmaras) source-prep wrappers for the DISTRIBUTED SA correct -----------------------
// Cell-local (gradU + grad(nuTilda) are the only halo-coupled inputs, supplied by the caller). Reuse the exact
// anon kernels + coeffs so the distributed nuTilda equation is byte-identical to the single-GPU physics.
void deviceSAStilda(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nuTilda,
    const DeviceBuffer<scalar>& y, scalar nu, const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& Stilda)
{
    const int nC = dm.nCells; Stilda.resize(nC);
    saStildaKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), nuTilda.data(), y.data(), nu, co, Stilda.data());
    cudaCheck(cudaGetLastError(), "deviceSAStilda");
}
void deviceSAFw(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, const DeviceBuffer<scalar>& Stilda,
    const DeviceBuffer<scalar>& y, const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& fw)
{
    const int nC = dm.nCells; fw.resize(nC);
    saFwKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), Stilda.data(), y.data(), co, fw.data());
    cudaCheck(cudaGetLastError(), "deviceSAFw");
}
void deviceSAMagSqr(const DeviceMesh& dm, const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& out)
{
    const int nC = dm.nCells; out.resize(nC);
    saMagSqrKernel<<<nBlocks(nC), TPB>>>(nC, gx.data(), gy.data(), gz.data(), out.data());
    cudaCheck(cudaGetLastError(), "deviceSAMagSqr");
}
void deviceSADEff(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, scalar nu, scalar sigmaNut, DeviceBuffer<scalar>& D)
{
    const int nC = dm.nCells; D.resize(nC);
    saDEffKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, sigmaNut, D.data());
    cudaCheck(cudaGetLastError(), "deviceSADEff");
}
void deviceSAReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, const DeviceBuffer<scalar>& Stilda,
    const DeviceBuffer<scalar>& fw, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& gradNt2,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src)
{
    saReactionKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.V.data(), nuTilda.data(), Stilda.data(),
        fw.data(), y.data(), gradNt2.data(), co, diag.data(), src.data());
    cudaCheck(cudaGetLastError(), "deviceSAReaction");
}

} // namespace brae
