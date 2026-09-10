// Maxwell viscoelastic laminar model -- the device kernels. See device_maxwell.cuh for the equations.
#include "device_maxwell.cuh"
#include "device_divdevreff.cuh"   // deviceBoundaryGradU / deviceTensorDivSource (shared with the stress path)
#include "pcuda_compat.cuh"

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// P = twoSymm(sigma & gradU) - nuM*rLambda*twoSymm(gradU).
//
// gradU is packed q = 3i + j = d(U_j)/d(x_i), so as a matrix G[i][j] = gradU[(3i+j)*nC + c] -- the same
// convention the rest of brae uses. (sigma & gradU)_ij = S_ik G_kj, and twoSymm(T) = T + T^T, so
//     P_ij = S_ik G_kj + S_jk G_ki - nuM*rLambda*(G_ij + G_ji)
// which is symmetric, hence six components rather than nine.
__device__
void maxwellPKernel(
    int nC,
    const scalar* __restrict__ sxx, const scalar* __restrict__ sxy, const scalar* __restrict__ sxz,
    const scalar* __restrict__ syy, const scalar* __restrict__ syz, const scalar* __restrict__ szz,
    const scalar* __restrict__ gradU,
    scalar nuM, scalar rLambda,
    scalar* __restrict__ pxx, scalar* __restrict__ pxy, scalar* __restrict__ pxz,
    scalar* __restrict__ pyy, scalar* __restrict__ pyz, scalar* __restrict__ pzz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar G[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) G[i][j] = gradU[(3*i + j)*nC + c];

    const scalar S[3][3] = {{sxx[c], sxy[c], sxz[c]},
                            {sxy[c], syy[c], syz[c]},
                            {sxz[c], syz[c], szz[c]}};

    scalar SG[3][3];                                   // (sigma & gradU)_ij = S_ik G_kj
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            scalar s = 0;
            for (int k = 0; k < 3; ++k) s += S[i][k]*G[k][j];
            SG[i][j] = s;
        }

    const scalar a = nuM*rLambda;
    auto P = [&](int i, int j) { return SG[i][j] + SG[j][i] - a*(G[i][j] + G[j][i]); };

    pxx[c] = P(0,0);
    pxy[c] = P(0,1);
    pxz[c] = P(0,2);
    pyy[c] = P(1,1);
    pyz[c] = P(1,2);
    pzz[c] = P(2,2);
}

// diag += V*rLambda (fvm::Sp), source += V*P. Same V-weighted convention as every other reaction here.
__device__
void maxwellReactionKernel(int nC, const scalar* __restrict__ V, scalar rLambda,
                           const scalar* __restrict__ P,
                           scalar* __restrict__ diag, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    diag[c]   += V[c]*rLambda;
    source[c] += V[c]*P[c];
}

// Expand the six stored components into the packed 9-component layout the tensor-divergence kernel
// takes. Cheap, and it keeps ONE divergence implementation rather than a symmetric copy of it.
__device__
void symmExpandKernel(int n,
                      const scalar* __restrict__ sxx, const scalar* __restrict__ sxy, const scalar* __restrict__ sxz,
                      const scalar* __restrict__ syy, const scalar* __restrict__ syz, const scalar* __restrict__ szz,
                      scalar* __restrict__ T)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    T[0*n + i] = sxx[i];  T[1*n + i] = sxy[i];  T[2*n + i] = sxz[i];
    T[3*n + i] = sxy[i];  T[4*n + i] = syy[i];  T[5*n + i] = syz[i];
    T[6*n + i] = sxz[i];  T[7*n + i] = syz[i];  T[8*n + i] = szz[i];
}

__device__
void scaleTensorKernel(int n9, const scalar* __restrict__ in, scalar s, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n9) out[i] = s*in[i];
}

// OF magSqr(symmTensor): the off-diagonals appear twice in the full tensor, so they count twice.
__device__
void symmMagSqrKernel(int nC,
                      const scalar* __restrict__ sxx, const scalar* __restrict__ sxy, const scalar* __restrict__ sxz,
                      const scalar* __restrict__ syy, const scalar* __restrict__ syz, const scalar* __restrict__ szz,
                      scalar* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    out[c] = sxx[c]*sxx[c] + syy[c]*syy[c] + szz[c]*szz[c]
           + scalar(2)*(sxy[c]*sxy[c] + sxz[c]*sxz[c] + syz[c]*syz[c]);
}
} // namespace

void deviceMaxwellP(int nC, const DeviceBuffer<scalar>* sigma, const DeviceBuffer<scalar>& gradU,
                    scalar nuM, scalar rLambda, DeviceBuffer<scalar>* P)
{
    for (int k = 0; k < 6; ++k) P[k].resize(static_cast<std::size_t>(nC));
    const scalar *s0=sigma[0].data(),*s1=sigma[1].data(),*s2=sigma[2].data(),*s3=sigma[3].data(),*s4=sigma[4].data(),*s5=sigma[5].data();
    const scalar* gradUd = gradU.data();
    scalar *p0=P[0].data(),*p1=P[1].data(),*p2=P[2].data(),*p3=P[3].data(),*p4=P[4].data(),*p5=P[5].data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
        maxwellPKernel(nC, s0, s1, s2, s3, s4, s5, gradUd, nuM, rLambda, p0, p1, p2, p3, p4, p5);
    });
    cudaCheck(cudaGetLastError(), "maxwellP");
}

void deviceMaxwellReaction(const DeviceBuffer<scalar>& V, scalar rLambda, const DeviceBuffer<scalar>& Pc,
                           DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source)
{
    const int nC = static_cast<int>(V.size());
    const scalar* Vd = V.data(); const scalar* Pcd = Pc.data();
    scalar* diagd = diag.data(); scalar* sourced = source.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
        maxwellReactionKernel(nC, Vd, rLambda, Pcd, diagd, sourced);
    });
    cudaCheck(cudaGetLastError(), "maxwellReaction");
}

void deviceDivSymmTensor(const DeviceMesh& dm,
                         const DeviceBuffer<scalar>* sigCell, const DeviceBuffer<scalar>* sigBnd,
                         DeviceBuffer<scalar>& outX, DeviceBuffer<scalar>& outY, DeviceBuffer<scalar>& outZ)
{
    const int nC = dm.nCells, nB = dm.nBndFaces;
    DeviceBuffer<scalar> Tc(static_cast<std::size_t>(9)*nC), Tb(static_cast<std::size_t>(9)*nB);
    {
        const scalar *c0=sigCell[0].data(),*c1=sigCell[1].data(),*c2=sigCell[2].data(),*c3=sigCell[3].data(),*c4=sigCell[4].data(),*c5=sigCell[5].data();
        scalar* Tcd = Tc.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { symmExpandKernel(nC, c0, c1, c2, c3, c4, c5, Tcd); });
    }
    cudaCheck(cudaGetLastError(), "symmExpand cells");
    if (nB)
    {
        const scalar *b0=sigBnd[0].data(),*b1=sigBnd[1].data(),*b2=sigBnd[2].data(),*b3=sigBnd[3].data(),*b4=sigBnd[4].data(),*b5=sigBnd[5].data();
        scalar* Tbd = Tb.data();
        pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () { symmExpandKernel(nB, b0, b1, b2, b3, b4, b5, Tbd); });
        cudaCheck(cudaGetLastError(), "symmExpand boundary");
    }
    deviceTensorDivSource(dm, Tc, Tb, outX, outY, outZ);
}

void deviceDivNuMGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                       const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                       const DeviceBuffer<scalar>& gradU, scalar nuM,
                       DeviceBuffer<scalar>& outX, DeviceBuffer<scalar>& outY, DeviceBuffer<scalar>& outZ)
{
    const int nC = dm.nCells, nB = dm.nBndFaces;
    DeviceBuffer<scalar> gradB;
    deviceBoundaryGradU(dm, dbU, Ux, Uy, Uz, gradU, gradB);
    DeviceBuffer<scalar> Tc(static_cast<std::size_t>(9)*nC), Tb(static_cast<std::size_t>(9)*nB);
    {
        const scalar* gradUd = gradU.data();
        scalar* Tcd = Tc.data();
        pcudaParallelFor(nBlocks(9*nC), TPB, [=] __device__ () { scaleTensorKernel(9*nC, gradUd, nuM, Tcd); });
    }
    cudaCheck(cudaGetLastError(), "scale nuM gradU cells");
    if (nB)
    {
        const scalar* gradBd = gradB.data();
        scalar* Tbd = Tb.data();
        pcudaParallelFor(nBlocks(9*nB), TPB, [=] __device__ () { scaleTensorKernel(9*nB, gradBd, nuM, Tbd); });
        cudaCheck(cudaGetLastError(), "scale nuM gradU boundary");
    }
    deviceTensorDivSource(dm, Tc, Tb, outX, outY, outZ);
}

void deviceSymmMagSqr(int nC, const DeviceBuffer<scalar>* sigma, DeviceBuffer<scalar>& out)
{
    out.resize(static_cast<std::size_t>(nC));
    const scalar *s0=sigma[0].data(),*s1=sigma[1].data(),*s2=sigma[2].data(),*s3=sigma[3].data(),*s4=sigma[4].data(),*s5=sigma[5].data();
    scalar* outd = out.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { symmMagSqrKernel(nC, s0, s1, s2, s3, s4, s5, outd); });
    cudaCheck(cudaGetLastError(), "symmMagSqr");
}

} // namespace brae
