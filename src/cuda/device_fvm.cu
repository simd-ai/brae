// cf GPU offload (G4): fvm matrix assembly on device. Per-face coefficients (no race), then a unified
// per-cell diag gather diag[c] = -(sum of lower over faces OWNED by c + sum of upper over faces
// NEIGHBOURing c), exactly the CPU diag[own]-=..., diag[nei]-=... scatter, turned into a gather.
//   laplacian : upper=lower = deltaCoeffs * gammaf * |Sf|
//   div upwind: lower = -max(phi,0)... w=(phi>=0); lower=-w*phi; upper=lower+phi
#include "device_mesh.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__device__
void lapFaceKernel(
    int nIf,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ gammaf,
    const scalar* __restrict__ magSf,
    scalar* __restrict__ upper,
    scalar* __restrict__ lower)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf)
    {
        const scalar c = dc[f] * gammaf[f] * magSf[f];
        upper[f] = c;
        lower[f] = c;
    }
}


// Gauss LINEAR (central differencing) -- OF's `linear` surfaceInterpolationScheme returns the plain
// geometric weights, mesh().surfaceInterpolation::weights() (linear.H:106). gaussConvectionScheme::fvmDiv
// then builds the SAME coefficients as upwind, only with those weights instead of pos0(phi):
//     lower = -w*phi ;  upper = lower + phi
// So this differs from divFaceKernel by one line -- the weight -- and nothing else. It is UNBOUNDED by
// construction (that is what central differencing is), which is why OF pairs it with `bounded` on
// convection-dominated cases and why LES uses it deliberately.
__device__
void divFaceLinearKernel(int nIf, const scalar* __restrict__ phi, const scalar* __restrict__ w,
                         scalar* __restrict__ upper, scalar* __restrict__ lower)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf)
    {
        const scalar p = phi[f];
        const scalar lo = -w[f] * p;
        lower[f] = lo;
        upper[f] = lo + p;
    }
}

__device__
void divFaceKernel(int nIf, const scalar* __restrict__ phi, scalar* __restrict__ upper, scalar* __restrict__ lower)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf)
    {
        const scalar p = phi[f];
        const scalar w = (p >= 0.0) ? 1.0 : 0.0;
        const scalar lo = -w * p;
        lower[f] = lo;
        upper[f] = lo + p;
    }
}


// limitedLinear convection: W_f = limiter*CDweight + (1-limiter)*pos0(phi); limiter = clamp(twoByk*r,0,1),
// r = NVDTVD gradient ratio. Reduces EXACTLY to upwind divFaceKernel at limiter=0. (OF gaussConvectionScheme +
// limitedSurfaceInterpolationScheme::weights + NVDTVD::r.) gradc{X,Y,Z} = grad(field); d = (Cf-C_own)-(Cf-C_nei).
__device__
void divLimitedFaceKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ cdw,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ field,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    scalar twoByk,
    scalar* __restrict__ upper,
    scalar* __restrict__ lower)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;

    const int P = own[f], N = nei[f];
    const scalar p = phi[f];
    const scalar dx = dOwnX[f] - dNeiX[f], dy = dOwnY[f] - dNeiY[f], dz = dOwnZ[f] - dNeiZ[f];   // d = C[N]-C[P]
    // NVDTVD::r, upwind-cell gradient (strict phi>0) projected on d, vs the face gradient.
    const int U = (p > 0.0) ? P : N;
    const scalar gradcf = dx*gx[U] + dy*gy[U] + dz*gz[U];
    const scalar gradf  = field[N] - field[P];
    scalar r;   // sign(s) = (s>=0)?1:-1  (OF Scalar.H)
    if (fabs(gradcf) >= 1000.0 * fabs(gradf))
        r = 2.0 * 1000.0 * ((gradcf >= 0.0) ? 1.0 : -1.0) * ((gradf >= 0.0) ? 1.0 : -1.0) - 1.0;
    else
        r = 2.0 * (gradcf / gradf) - 1.0;
    // twoByk > 0 selects limitedLinear (limiter = clamp(2/k * r, 0, 1)); twoByk == 0 selects vanAlbada
    // (limiter = r(r+1)/(r^2+1), vanAlbada.H:85), which the Maxwell tutorials name for div(phi,sigma).
    // Same NVDTVD r either way -- only the limiter function differs, so they share one kernel.
    scalar limiter;
    if (twoByk > 0.0)
    {
        limiter = twoByk * r;
        limiter = (limiter < 0.0) ? 0.0 : (limiter > 1.0 ? 1.0 : limiter);    // clamp(.,0,1)
    }
    else
    {
        limiter = r * (r + 1.0) / (r*r + 1.0);        // OF vanAlbada: NOT clamped, and it is <= 1 anyway
    }
    const scalar pos0 = (p >= 0.0) ? 1.0 : 0.0;
    const scalar W = limiter * cdw[f] + (1.0 - limiter) * pos0;
    const scalar lo = -W * p;
    lower[f] = lo;
    upper[f] = lo + p;
}


// limitedLinearV convection: the OF "V" scheme (NVDVTVDV::r + null LimitFunc) -- a SINGLE limiter per face computed
// from the VECTOR field (not per component, not magSqr): gradfV = U[N]-U[P], gradf = gradfV.gradfV,
// gradcf = gradfV.(d & gradU[upwind]), r = 2*gradcf/gradf - 1. One limiter -> W applied to all 3 components (the
// convection matrix is shared, so this feeds mDiag/mUp/mLo implicitly, exactly as the magSqr path does). gU{n}{x,y,z}
// = grad(U_n) cell fields; d = (Cf-C_own)-(Cf-C_nei) = C[N]-C[P].
__device__
void divLimitedVFaceKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ cdw,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ U0,
    const scalar* __restrict__ U1,
    const scalar* __restrict__ U2,
    const scalar* __restrict__ gU0x,
    const scalar* __restrict__ gU0y,
    const scalar* __restrict__ gU0z,
    const scalar* __restrict__ gU1x,
    const scalar* __restrict__ gU1y,
    const scalar* __restrict__ gU1z,
    const scalar* __restrict__ gU2x,
    const scalar* __restrict__ gU2y,
    const scalar* __restrict__ gU2z,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    scalar twoByk,
    scalar* __restrict__ upper,
    scalar* __restrict__ lower)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;

    const int P = own[f], N = nei[f];
    const scalar p = phi[f];
    const scalar dx = dOwnX[f] - dNeiX[f], dy = dOwnY[f] - dNeiY[f], dz = dOwnZ[f] - dNeiZ[f];   // d = C[N]-C[P]
    const scalar g0 = U0[N] - U0[P], g1 = U1[N] - U1[P], g2 = U2[N] - U2[P];   // gradfV = U[N]-U[P]
    const scalar gradf = g0*g0 + g1*g1 + g2*g2;                                // gradfV . gradfV
    const int U = (p > 0.0) ? P : N;                                          // strict upwind cell
    const scalar dgU0 = dx*gU0x[U] + dy*gU0y[U] + dz*gU0z[U];                  // (d & gradU)[k] at the upwind cell
    const scalar dgU1 = dx*gU1x[U] + dy*gU1y[U] + dz*gU1z[U];
    const scalar dgU2 = dx*gU2x[U] + dy*gU2y[U] + dz*gU2z[U];
    const scalar gradcf = g0*dgU0 + g1*dgU1 + g2*dgU2;                         // gradfV . (d & gradU)
    scalar r;
    if (fabs(gradcf) >= 1000.0 * fabs(gradf))
        r = 2.0 * 1000.0 * ((gradcf >= 0.0) ? 1.0 : -1.0) * ((gradf >= 0.0) ? 1.0 : -1.0) - 1.0;
    else
        r = 2.0 * (gradcf / gradf) - 1.0;
    scalar limiter = twoByk * r;
    limiter = (limiter < 0.0) ? 0.0 : (limiter > 1.0 ? 1.0 : limiter);        // clamp(.,0,1)
    const scalar pos0 = (p >= 0.0) ? 1.0 : 0.0;
    const scalar W = limiter * cdw[f] + (1.0 - limiter) * pos0;
    const scalar lo = -W * p;
    lower[f] = lo;
    upper[f] = lo + p;
}


__device__
void diagGatherKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    scalar* __restrict__ diag)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
        s += lower[f];              // faces owned by c
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
        s += upper[losort[k]];    // faces neighbouring c
    diag[c] = -s;
}


// Fold the boundary into the solve: diagC = rawDiag + sum internalCoeffs over c's boundary faces;
// b = V*divPhi + sum boundaryCoeffs (= brae::pcg's addBoundaryDiag/addBoundarySource + pEqn source V*div).
__device__
void foldPressureKernel(
    int nC,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ rawDiag,
    const scalar* __restrict__ V,
    const scalar* __restrict__ divPhi,
    const scalar* __restrict__ iC,
    const scalar* __restrict__ bC,
    scalar* __restrict__ diagC,
    scalar* __restrict__ b)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar dd = rawDiag[c], bb = V[c] * divPhi[c];
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int kk = bndPerm[k];
        dd += iC[kk];
        bb += bC[kk];
    }
    diagC[c] = dd;
    b[c] = bb;
}
} // namespace


void deviceLaplacianCoeffs(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& gammafInt,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& upper,
    DeviceBuffer<scalar>& lower,
    bool nonOrth)
{
    const int nIf = dm.nInternalFaces, nC = dm.nCells;
    upper.resize(nIf);
    lower.resize(nIf);
    diag.resize(nC);
    // corrected scheme uses nonOrthDeltaCoeffs = 1/max(n.delta, 0.05|delta|) for the implicit part.
    const scalar* dcPtr = nonOrth ? dm.nonOrthDc.data() : dm.dc.data();
    {
        const scalar* gammafD = gammafInt.data(); const scalar* magSf = dm.magSf.data();
        scalar* upperD = upper.data(); scalar* lowerD = lower.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            lapFaceKernel(nIf, dcPtr, gammafD, magSf, upperD, lowerD); });
    }
    cudaCheck(cudaGetLastError(), "lapFace");
    {
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const scalar* upperD = upper.data(); const scalar* lowerD = lower.data(); scalar* diagD = diag.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            diagGatherKernel(nC, ownerStart, losort, losortStart, upperD, lowerD, diagD); });
    }
    cudaCheck(cudaGetLastError(), "diagGather");
}


namespace {
// faceFluxCorr_f = gamma_f*|Sf|_f * (corrVec_f . grad(vf)_f), grad linearly interpolated owner/neighbour.
__device__
void lapCorrFaceKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gammaf,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    scalar* __restrict__ ffc)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;

    const int o = own[f], n = nei[f];
    const scalar wf = w[f], wn = 1.0 - wf;
    const scalar gfx = wf*gx[o] + wn*gx[n], gfy = wf*gy[o] + wn*gy[n], gfz = wf*gz[o] + wn*gz[n];
    ffc[f] = gammaf[f] * magSf[f] * (cvx[f]*gfx + cvy[f]*gfy + cvz[f]*gfz);
}


// limitedSnGrad variant (OpenFOAM fv::limitedSnGrad). Same correction flux as lapCorrFaceKernel but per-face scaled
// by the OF limiter so the explicit correction never exceeds (psi/(1-psi)) times the orthogonal contribution:
//   corr_f   = corrVec_f . grad(p)_f                         (the snGrad correction)
//   orthSn_f = nonOrthDc_f * (p_N - p_P)                     (orthogonal snGrad, over-relaxed deltaCoeffs)
//   limiter_f= min( psi*|orthSn_f| / ((1-psi)*|corr_f| + SMALL), 1 )
//   ffc_f    = gamma_f*|Sf|_f * limiter_f * corr_f
// psi=1 -> unlimited (== corrected); the caller skips this kernel entirely then (bit-identical). psi<1 caps only the
// pathological faces where |corr| > |orthSn| (e.g. extreme-aspect-ratio + high-non-orth cells); well-behaved faces
// (|corr| small) keep limiter==1 == full correction, so accuracy on the bulk mesh is preserved.
__device__
void lapCorrFaceLimitedKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gammaf,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ nonOrthDc,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    scalar psi,
    scalar* __restrict__ ffc)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;

    const int o = own[f], n = nei[f];
    const scalar wf = w[f], wn = 1.0 - wf;
    const scalar gfx = wf*gx[o] + wn*gx[n], gfy = wf*gy[o] + wn*gy[n], gfz = wf*gz[o] + wn*gz[n];
    const scalar corr   = cvx[f]*gfx + cvy[f]*gfy + cvz[f]*gfz;     // snGrad correction
    const scalar orthSn = nonOrthDc[f] * (phi[n] - phi[o]);        // orthogonal snGrad (over-relaxed)
    scalar limiter = (psi * fabs(orthSn)) / ((1.0 - psi) * fabs(corr) + 1.0e-15);
    if (limiter > 1.0) limiter = 1.0;
    ffc[f] = gammaf[f] * magSf[f] * limiter * corr;
}


// lapCorrSource[c] = -V*fvc::div(ffc)[c] = -sum_{f: owner=c} ffc + sum_{f: nei=c} ffc.
__device__
void lapCorrGatherKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ ffc,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
        s -= ffc[f];               // c is owner
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
        s += ffc[losort[k]];      // c is neighbour
    src[c] = s;
}
} // namespace


// Face flux correction ffc_f = gamma_f*|Sf|_f*(corrVec_f . grad(vf)_f), the OF faceFluxCorrection (isotropic gamma).
void deviceLaplacianCorrFlux(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& gammafInt,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& ffc)
{
    const int nIf = dm.nInternalFaces;
    ffc.resize(nIf);
    {
        const label* own = dm.owner.data(); const label* nei = dm.nei.data(); const scalar* w = dm.w.data();
        const scalar* gammafD = gammafInt.data(); const scalar* magSf = dm.magSf.data();
        const scalar* cvx = dm.corrVecX.data(); const scalar* cvy = dm.corrVecY.data(); const scalar* cvz = dm.corrVecZ.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        scalar* ffcd = ffc.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            lapCorrFaceKernel(nIf, own, nei, w, gammafD, magSf, cvx, cvy, cvz, gxd, gyd, gzd, ffcd); });
    }
    cudaCheck(cudaGetLastError(), "lapCorrFace");
}


// limitedSnGrad correction flux: per-face OF limiter (psi in [0,1]). psi>=1 falls back to the unlimited path above.
// phi = the scalar field being corrected (for the orthogonal snGrad reference).
void deviceLaplacianCorrFluxLimited(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& gammafInt,
    const DeviceBuffer<scalar>& phi,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar psi,
    DeviceBuffer<scalar>& ffc)
{
    if (psi >= 1.0)
    {
        deviceLaplacianCorrFlux(dm, gammafInt, gx, gy, gz, ffc);
        return;
    }
    const int nIf = dm.nInternalFaces;
    ffc.resize(nIf);
    {
        const label* own = dm.owner.data(); const label* nei = dm.nei.data(); const scalar* w = dm.w.data();
        const scalar* gammafD = gammafInt.data(); const scalar* magSf = dm.magSf.data();
        const scalar* cvx = dm.corrVecX.data(); const scalar* cvy = dm.corrVecY.data(); const scalar* cvz = dm.corrVecZ.data();
        const scalar* nonOrthDc = dm.nonOrthDc.data(); const scalar* phid = phi.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        scalar* ffcd = ffc.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            lapCorrFaceLimitedKernel(nIf, own, nei, w, gammafD, magSf, cvx, cvy, cvz, nonOrthDc, phid, gxd, gyd, gzd, psi, ffcd); });
    }
    cudaCheck(cudaGetLastError(), "lapCorrFaceLimited");
}


// src = -V*fvc::div(ffc) (the integrated face-flux divergence, = fvm::laplacian.source()'s correction term).
void deviceFaceDivSource(const DeviceMesh& dm, const DeviceBuffer<scalar>& ffc, DeviceBuffer<scalar>& src)
{
    const int nC = dm.nCells;
    src.resize(nC);
    {
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const scalar* ffcD = ffc.data(); scalar* srcd = src.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            lapCorrGatherKernel(nC, ownerStart, losort, losortStart, ffcD, srcd); });
    }
    cudaCheck(cudaGetLastError(), "lapCorrGather");
}


void deviceLaplacianCorr(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& gammafInt,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& lapCorrSource)
{
    DeviceBuffer<scalar> ffc;
    deviceLaplacianCorrFlux(dm, gammafInt, gx, gy, gz, ffc);
    deviceFaceDivSource(dm, ffc, lapCorrSource);
}


// The central-difference counterpart of deviceDivUpwindCoeffs, sharing its diagonal gather.
void deviceDivCentralCoeffs(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& upper,
    DeviceBuffer<scalar>& lower)
{
    const int nIf = dm.nInternalFaces, nC = dm.nCells;
    upper.resize(nIf);
    lower.resize(nIf);
    diag.resize(nC);
    {
        const scalar* phiIntD = phiInt.data(); const scalar* w = dm.w.data();
        scalar* upperD = upper.data(); scalar* lowerD = lower.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            divFaceLinearKernel(nIf, phiIntD, w, upperD, lowerD); });
    }
    cudaCheck(cudaGetLastError(), "divFaceLinear");
    {
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const scalar* upperD = upper.data(); const scalar* lowerD = lower.data(); scalar* diagD = diag.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            diagGatherKernel(nC, ownerStart, losort, losortStart, upperD, lowerD, diagD); });
    }
    cudaCheck(cudaGetLastError(), "diagGather");
}

void deviceDivUpwindCoeffs(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& upper,
    DeviceBuffer<scalar>& lower)
{
    const int nIf = dm.nInternalFaces, nC = dm.nCells;
    upper.resize(nIf);
    lower.resize(nIf);
    diag.resize(nC);
    {
        const scalar* phiIntD = phiInt.data(); scalar* upperD = upper.data(); scalar* lowerD = lower.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            divFaceKernel(nIf, phiIntD, upperD, lowerD); });
    }
    cudaCheck(cudaGetLastError(), "divFace");
    {
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const scalar* upperD = upper.data(); const scalar* lowerD = lower.data(); scalar* diagD = diag.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            diagGatherKernel(nC, ownerStart, losort, losortStart, upperD, lowerD, diagD); });
    }
    cudaCheck(cudaGetLastError(), "diagGather");
}


void deviceDivLimitedCoeffs(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar twoByk,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& upper,
    DeviceBuffer<scalar>& lower)
{
    const int nIf = dm.nInternalFaces, nC = dm.nCells;
    upper.resize(nIf);
    lower.resize(nIf);
    diag.resize(nC);
    {
        const label* own = dm.owner.data(); const label* nei = dm.nei.data(); const scalar* w = dm.w.data();
        const scalar* phiIntD = phiInt.data(); const scalar* fieldD = field.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        const scalar* dOwnX = dm.dOwnX.data(); const scalar* dOwnY = dm.dOwnY.data(); const scalar* dOwnZ = dm.dOwnZ.data();
        const scalar* dNeiX = dm.dNeiX.data(); const scalar* dNeiY = dm.dNeiY.data(); const scalar* dNeiZ = dm.dNeiZ.data();
        scalar* upperD = upper.data(); scalar* lowerD = lower.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            divLimitedFaceKernel(nIf, own, nei, w, phiIntD, fieldD, gxd, gyd, gzd,
                                  dOwnX, dOwnY, dOwnZ, dNeiX, dNeiY, dNeiZ, twoByk, upperD, lowerD); });
    }
    cudaCheck(cudaGetLastError(), "divLimitedFace");
    {
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const scalar* upperD = upper.data(); const scalar* lowerD = lower.data(); scalar* diagD = diag.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            diagGatherKernel(nC, ownerStart, losort, losortStart, upperD, lowerD, diagD); });
    }
    cudaCheck(cudaGetLastError(), "diagGather");
}


void deviceDivLimitedVCoeffs(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>* U,       // U[3]     component cell fields
    const DeviceBuffer<scalar>* gUx,     // gUx[3]   d(U_n)/dx
    const DeviceBuffer<scalar>* gUy,     // gUy[3]   d(U_n)/dy
    const DeviceBuffer<scalar>* gUz,     // gUz[3]   d(U_n)/dz
    scalar twoByk,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& upper,
    DeviceBuffer<scalar>& lower)
{
    const int nIf = dm.nInternalFaces, nC = dm.nCells;
    upper.resize(nIf);
    lower.resize(nIf);
    diag.resize(nC);
    {
        const label* own = dm.owner.data(); const label* nei = dm.nei.data(); const scalar* w = dm.w.data();
        const scalar* phiIntD = phiInt.data();
        const scalar* U0 = U[0].data(); const scalar* U1 = U[1].data(); const scalar* U2 = U[2].data();
        const scalar* gU0x = gUx[0].data(); const scalar* gU0y = gUy[0].data(); const scalar* gU0z = gUz[0].data();
        const scalar* gU1x = gUx[1].data(); const scalar* gU1y = gUy[1].data(); const scalar* gU1z = gUz[1].data();
        const scalar* gU2x = gUx[2].data(); const scalar* gU2y = gUy[2].data(); const scalar* gU2z = gUz[2].data();
        const scalar* dOwnX = dm.dOwnX.data(); const scalar* dOwnY = dm.dOwnY.data(); const scalar* dOwnZ = dm.dOwnZ.data();
        const scalar* dNeiX = dm.dNeiX.data(); const scalar* dNeiY = dm.dNeiY.data(); const scalar* dNeiZ = dm.dNeiZ.data();
        scalar* upperD = upper.data(); scalar* lowerD = lower.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            divLimitedVFaceKernel(nIf, own, nei, w, phiIntD, U0, U1, U2,
                                   gU0x, gU0y, gU0z, gU1x, gU1y, gU1z, gU2x, gU2y, gU2z,
                                   dOwnX, dOwnY, dOwnZ, dNeiX, dNeiY, dNeiZ, twoByk, upperD, lowerD); });
    }
    cudaCheck(cudaGetLastError(), "divLimitedVFace");
    {
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const scalar* upperD = upper.data(); const scalar* lowerD = lower.data(); scalar* diagD = diag.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            diagGatherKernel(nC, ownerStart, losort, losortStart, upperD, lowerD, diagD); });
    }
    cudaCheck(cudaGetLastError(), "diagGather");
}


// Same scheme, taking the PACKED 9-component cell gradient deviceGradU produces (gradU[q*nC + c],
// q = 3i + j = d(U_j)/d(x_i)) instead of nine separate buffers -- so the caller can reuse the one
// gradient it already limits with deviceCellLimitGradU, which is the gradient OF's LimitedScheme
// takes (fvc::grad(phi) -> the gradSchemes `grad(U)` entry).
void deviceDivLimitedVCoeffs(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>* U,
    const DeviceBuffer<scalar>& gradU,
    scalar twoByk,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& upper,
    DeviceBuffer<scalar>& lower)
{
    const int nIf = dm.nInternalFaces, nC = dm.nCells;
    upper.resize(nIf);
    lower.resize(nIf);
    diag.resize(nC);
    const scalar* g = gradU.data();
    {
        const label* own = dm.owner.data(); const label* nei = dm.nei.data(); const scalar* w = dm.w.data();
        const scalar* phiIntD = phiInt.data();
        const scalar* U0 = U[0].data(); const scalar* U1 = U[1].data(); const scalar* U2 = U[2].data();
        const scalar* gU0x = g + 0*nC; const scalar* gU0y = g + 3*nC; const scalar* gU0z = g + 6*nC;
        const scalar* gU1x = g + 1*nC; const scalar* gU1y = g + 4*nC; const scalar* gU1z = g + 7*nC;
        const scalar* gU2x = g + 2*nC; const scalar* gU2y = g + 5*nC; const scalar* gU2z = g + 8*nC;
        const scalar* dOwnX = dm.dOwnX.data(); const scalar* dOwnY = dm.dOwnY.data(); const scalar* dOwnZ = dm.dOwnZ.data();
        const scalar* dNeiX = dm.dNeiX.data(); const scalar* dNeiY = dm.dNeiY.data(); const scalar* dNeiZ = dm.dNeiZ.data();
        scalar* upperD = upper.data(); scalar* lowerD = lower.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            divLimitedVFaceKernel(nIf, own, nei, w, phiIntD, U0, U1, U2,
                                   gU0x, gU0y, gU0z, gU1x, gU1y, gU1z, gU2x, gU2y, gU2z,
                                   dOwnX, dOwnY, dOwnZ, dNeiX, dNeiY, dNeiZ, twoByk, upperD, lowerD); });
    }
    cudaCheck(cudaGetLastError(), "divLimitedVFacePacked");
    {
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const scalar* upperD = upper.data(); const scalar* lowerD = lower.data(); scalar* diagD = diag.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            diagGatherKernel(nC, ownerStart, losort, losortStart, upperD, lowerD, diagD); });
    }
    cudaCheck(cudaGetLastError(), "diagGather");
}


void deviceFoldPressure(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& rawDiag,
    const DeviceBuffer<scalar>& divPhi,
    const DeviceBuffer<scalar>& iC,
    const DeviceBuffer<scalar>& bC,
    DeviceBuffer<scalar>& diagC,
    DeviceBuffer<scalar>& b)
{
    const int nC = dm.nCells;
    diagC.resize(nC);
    b.resize(nC);
    {
        const label* bndCellStart = dm.bndCellStart.data(); const label* bndPerm = dm.bndPerm.data();
        const scalar* rawDiagD = rawDiag.data(); const scalar* Vd = dm.V.data(); const scalar* divPhiD = divPhi.data();
        const scalar* iCd = iC.data(); const scalar* bCd = bC.data();
        scalar* diagCd = diagC.data(); scalar* bd = b.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            foldPressureKernel(nC, bndCellStart, bndPerm, rawDiagD, Vd, divPhiD, iCd, bCd, diagCd, bd); });
    }
    cudaCheck(cudaGetLastError(), "foldPressure");
}


namespace {
__device__
void foldKernel(
    int nC,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ rawDiag,
    const scalar* __restrict__ source,
    const scalar* __restrict__ iC,
    const scalar* __restrict__ bC,
    scalar* __restrict__ diagC,
    scalar* __restrict__ b)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar dd = rawDiag[c], bb = source[c];
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int kk = bndPerm[k];
        dd += iC[kk];
        bb += bC[kk];
    }
    diagC[c] = dd;
    b[c] = bb;
}
} // namespace


// Generic boundary fold: diagC = rawDiag + sum internalCoeffs; b = source + sum boundaryCoeffs (source given directly).
void deviceFold(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& rawDiag,
    const DeviceBuffer<scalar>& source,
    const DeviceBuffer<scalar>& iC,
    const DeviceBuffer<scalar>& bC,
    DeviceBuffer<scalar>& diagC,
    DeviceBuffer<scalar>& b)
{
    const int nC = dm.nCells;
    diagC.resize(nC);
    b.resize(nC);
    {
        const label* bndCellStart = dm.bndCellStart.data(); const label* bndPerm = dm.bndPerm.data();
        const scalar* rawDiagD = rawDiag.data(); const scalar* sourceD = source.data();
        const scalar* iCd = iC.data(); const scalar* bCd = bC.data();
        scalar* diagCd = diagC.data(); scalar* bd = b.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            foldKernel(nC, bndCellStart, bndPerm, rawDiagD, sourceD, iCd, bCd, diagCd, bd); });
    }
    cudaCheck(cudaGetLastError(), "fold");
}

} // namespace brae
