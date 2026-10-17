// DEShybrid convection scheme (OF TurbulenceModels/schemes/DEShybrid), transcribed from
// DEShybrid.H::calcBlendingFactor. See deshybrid_coeffs.cuh for the formula and the reduction to a
// deferred correction against brae's upwind matrix.
#include "device_deshybrid.cuh"
#include "device_scalar_transport.cuh"   // nBlocks/TPB (shared launch geometry)
#include <cuda_runtime.h>

namespace brae {

namespace {

__global__
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
__global__
void desFaceCorrKernel(
    int nIf, const label* __restrict__ own, const label* __restrict__ nei,
    const scalar* __restrict__ phi, const scalar* __restrict__ w, const scalar* __restrict__ sigma,
    const scalar* __restrict__ field, const scalar* __restrict__ gx, const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    FaceOffsets fo,
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
    const vector d = fo.of(f, up);
    const scalar dx = d.x;
    const scalar dy = d.y;
    const scalar dz = d.z;
    const scalar lu = gx[up]*dx + gy[up]*dy + gz[up]*dz;

    fc[f] = (scalar(1) - bf)*lin + bf*lu;
}

__global__
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
    desSigmaKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), V.data(),
                                         nut.size() ? nut.data() : nullptr, nu, co,
                                         (delta && delta->size()) ? delta->data() : nullptr, sigma.data());
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
    if (nIf > 0)
        desFaceCorrKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), phiInt.data(),
            dm.w.data(), sigma.data(), field.data(), gx.data(), gy.data(), gz.data(),
            faceOffsets(dm), fc.data());
    desDivKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(),
        dm.losortStart.data(), phiInt.data(), fc.data(), corr.data());
    cudaCheck(cudaGetLastError(), "desHybridCorr");
}

} // namespace brae
