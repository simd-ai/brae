// DEShybrid convection scheme (OF TurbulenceModels/schemes/DEShybrid), transcribed from
// DEShybrid.H::calcBlendingFactor. See deshybrid_coeffs.cuh for the formula and the reduction to a
// deferred correction against brae's upwind matrix.
#include "device_deshybrid.cuh"
#include "device_scalar_transport.cuh"   // nBlocks/TPB (shared launch geometry)
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {

__device__
void desSigmaKernel(
    int nC, const scalar* __restrict__ gradU, const scalar* __restrict__ V,
    const scalar* __restrict__ nut, scalar nu, DesHybridCoeffs co,
    const scalar* __restrict__ dOpt, scalar* __restrict__ sigma)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q) t[q] = gradU[q * nC + c];

    // S = sqrt(2)*mag(symm(gradU)),  Omega = sqrt(2)*mag(skew(gradU)); OF's mag() of a tensor is the
    // Frobenius norm, so mag(symm) = sqrt(sum_ij S_ij^2).
    scalar ss = 0, ww = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar s = scalar(0.5)*(t[i*3+j] + t[j*3+i]);
            const scalar w = scalar(0.5)*(t[i*3+j] - t[j*3+i]);
            ss += s*s;
            ww += w*w;
        }
    const scalar S     = sqrt(scalar(2)*ss);
    const scalar Omega = sqrt(scalar(2)*ww);
    const scalar tau0  = co.L0 / co.U0;
    const scalar d     = dOpt ? dOpt[c] : cbrt(V[c]);   // the SAME filter width the LES model uses

    const scalar half  = scalar(0.5)*(S*S + Omega*Omega);
    const scalar oLim  = co.OmegaLim / tau0;
    const scalar B     = co.CH3 * Omega * fmax(S, Omega) / fmax(half, oLim*oLim);
    const scalar B4    = B*B*B*B;
    const scalar gB    = tanh(B4);
    const scalar K     = fmax(sqrt(half), scalar(0.1)/tau0);

    const scalar nutC  = nut ? nut[c] : scalar(0);
    const scalar Csd   = co.Cs * d;
    const scalar nuEff = fmax(nutC, fmin(Csd*Csd*S, co.nutLim*nutC)) + nu;
    const scalar lTurb = sqrt(fmax(nuEff / (pow(scalar(0.09), scalar(1.5)) * K), scalar(0)));

    // SMALL*L0 keeps the ratio finite where the sensor has switched fully off (g -> 0 in irrotational flow)
    const scalar A = co.CH2 * fmax(scalar(0), co.CDES*d / fmax(lTurb*gB, scalar(1e-15)*co.L0) - scalar(0.5));
    sigma[c] = fmax(co.sigmaMax * tanh(pow(A, co.CH1)), co.sigmaMin);
}

// Per FACE: blend the two schemes' own deferred corrections (relative to the upwind matrix brae builds),
// weighted by bf = interpolate(sigma).
//   linear      : linear_face - upwind_face = (w - pos0(phi))*(field[own] - field[nbr])
//   linearUpwind: grad[upwind] . (Cf - C[upwind])
__device__
void desFaceCorrKernel(
    int nIf, const label* __restrict__ own, const label* __restrict__ nei,
    const scalar* __restrict__ phi, const scalar* __restrict__ w, const scalar* __restrict__ sigma,
    const scalar* __restrict__ field, const scalar* __restrict__ gx, const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    scalar* __restrict__ fc)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const int o = own[f], n = nei[f];
    const scalar pf = phi[f];
    const scalar bf = w[f]*sigma[o] + (scalar(1) - w[f])*sigma[n];   // fvc::interpolate(sigma)

    const scalar pos0 = (pf >= 0) ? scalar(1) : scalar(0);
    const scalar lin  = (w[f] - pos0) * (field[o] - field[n]);

    const int up = (pf >= 0) ? o : n;
    const scalar dx = (pf >= 0) ? dOwnX[f] : dNeiX[f];
    const scalar dy = (pf >= 0) ? dOwnY[f] : dNeiY[f];
    const scalar dz = (pf >= 0) ? dOwnZ[f] : dNeiZ[f];
    const scalar lu = gx[up]*dx + gy[up]*dy + gz[up]*dz;

    fc[f] = (scalar(1) - bf)*lin + bf*lu;
}

__device__
void desDivKernel(
    int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
    const label* __restrict__ losortStart, const scalar* __restrict__ phi,
    const scalar* __restrict__ fc, scalar* __restrict__ corr)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) s += phi[f] * fc[f];
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    { const int f = losort[k]; s -= phi[f] * fc[f]; }
    corr[c] = s;
}

} // namespace


void deviceDesHybridSigma(int nC, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& V,
                          const DeviceBuffer<scalar>& nut, scalar nu, const DesHybridCoeffs& co,
                          DeviceBuffer<scalar>& sigma, const DeviceBuffer<scalar>* delta)
{
    sigma.resize(nC);
    if (nC == 0) return;
    const scalar* gradUd = gradU.data();
    const scalar* Vd = V.data();
    const scalar* nutd = nut.size() ? nut.data() : nullptr;
    const scalar* deltad = (delta && delta->size()) ? delta->data() : nullptr;
    scalar* sigmad = sigma.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
        desSigmaKernel(nC, gradUd, Vd, nutd, nu, co, deltad, sigmad);
    });
    cudaCheck(cudaGetLastError(), "desHybridSigma");
}


void deviceDesHybridCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt,
                         const DeviceBuffer<scalar>& sigma, const DeviceBuffer<scalar>& field,
                         const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                         const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corr)
{
    const int nIf = dm.nInternalFaces;
    corr.resize(dm.nCells);
    if (dm.nCells == 0) return;
    DeviceBuffer<scalar> fc(nIf);
    const scalar* phiIntd = phiInt.data();
    scalar* fcd = fc.data();
    if (nIf > 0)
    {
        const label* ownerd = dm.owner.data(); const label* neid = dm.nei.data();
        const scalar* wd = dm.w.data(); const scalar* sigmad = sigma.data(); const scalar* fieldd = field.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        const scalar* dOwnXd = dm.dOwnX.data(); const scalar* dOwnYd = dm.dOwnY.data(); const scalar* dOwnZd = dm.dOwnZ.data();
        const scalar* dNeiXd = dm.dNeiX.data(); const scalar* dNeiYd = dm.dNeiY.data(); const scalar* dNeiZd = dm.dNeiZ.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            desFaceCorrKernel(nIf, ownerd, neid, phiIntd, wd, sigmad, fieldd, gxd, gyd, gzd,
                              dOwnXd, dOwnYd, dOwnZd, dNeiXd, dNeiYd, dNeiZd, fcd);
        });
    }
    const int nCells = dm.nCells;
    const label* ownerStartd = dm.ownerStart.data(); const label* losortd = dm.losort.data(); const label* losortStartd = dm.losortStart.data();
    scalar* corrd = corr.data();
    pcudaParallelFor(nBlocks(nCells), TPB, [=] __device__ () {
        desDivKernel(nCells, ownerStartd, losortd, losortStartd, phiIntd, fcd, corrd);
    });
    cudaCheck(cudaGetLastError(), "desHybridCorr");
}

} // namespace brae
