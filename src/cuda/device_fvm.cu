// cf GPU offload (G4): fvm matrix assembly on device. Per-face coefficients (no race), then a unified
// per-cell diag gather diag[c] = -(sum of lower over faces OWNED by c + sum of upper over faces
// NEIGHBOURing c), exactly the CPU diag[own]-=..., diag[nei]-=... scatter, turned into a gather.
//   laplacian : upper=lower = deltaCoeffs * gammaf * |Sf|
//   div upwind: lower = -max(phi,0)... w=(phi>=0); lower=-w*phi; upper=lower+phi
#include "device_mesh.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
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
__global__
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

__global__
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
__global__
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
__global__
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


__global__
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
__global__
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
    lapFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dcPtr, gammafInt.data(), dm.magSf.data(), upper.data(), lower.data());
    cudaCheck(cudaGetLastError(), "lapFace");
    diagGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), upper.data(), lower.data(), diag.data());
    cudaCheck(cudaGetLastError(), "diagGather");
}


namespace {
// faceFluxCorr_f = gamma_f*|Sf|_f * (corrVec_f . grad(vf)_f), grad linearly interpolated owner/neighbour.
__global__
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
__global__
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
__global__
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
    lapCorrFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), gammafInt.data(), dm.magSf.data(),
                                             dm.corrVecX.data(), dm.corrVecY.data(), dm.corrVecZ.data(),
                                             gx.data(), gy.data(), gz.data(), ffc.data());
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
    lapCorrFaceLimitedKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), gammafInt.data(), dm.magSf.data(),
                                                    dm.corrVecX.data(), dm.corrVecY.data(), dm.corrVecZ.data(),
                                                    dm.nonOrthDc.data(), phi.data(), gx.data(), gy.data(), gz.data(), psi, ffc.data());
    cudaCheck(cudaGetLastError(), "lapCorrFaceLimited");
}

namespace {
// The VECTOR form. OF's limitedSnGrad<Type> takes mag() of the whole snGrad and of the whole correction,
// so a vector field gets ONE limiter per face shared by all three components -- not three independent
// ones. Limiting per component is a different scheme: it lets one component's correction survive where
// the vector's own magnitude says the face should be capped, and on airFoil2D the two differ by 0.6%.
__global__
void lapCorrFaceLimitedVecKernel(
    int nIf,
    const label*  __restrict__ own,
    const label*  __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gammaf,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ nonOrthDc,
    const scalar* __restrict__ p0,
    const scalar* __restrict__ p1,
    const scalar* __restrict__ p2,
    const scalar* __restrict__ g0x, const scalar* __restrict__ g0y, const scalar* __restrict__ g0z,
    const scalar* __restrict__ g1x, const scalar* __restrict__ g1y, const scalar* __restrict__ g1z,
    const scalar* __restrict__ g2x, const scalar* __restrict__ g2y, const scalar* __restrict__ g2z,
    scalar psi,
    scalar* __restrict__ f0,
    scalar* __restrict__ f1,
    scalar* __restrict__ f2)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const int o = own[f], n = nei[f];
    const scalar wf = w[f], wn = 1.0 - wf;

    const scalar c0 = cvx[f]*(wf*g0x[o] + wn*g0x[n]) + cvy[f]*(wf*g0y[o] + wn*g0y[n])
                    + cvz[f]*(wf*g0z[o] + wn*g0z[n]);
    const scalar c1 = cvx[f]*(wf*g1x[o] + wn*g1x[n]) + cvy[f]*(wf*g1y[o] + wn*g1y[n])
                    + cvz[f]*(wf*g1z[o] + wn*g1z[n]);
    const scalar c2 = cvx[f]*(wf*g2x[o] + wn*g2x[n]) + cvy[f]*(wf*g2y[o] + wn*g2y[n])
                    + cvz[f]*(wf*g2z[o] + wn*g2z[n]);

    const scalar d0 = nonOrthDc[f]*(p0[n] - p0[o]);
    const scalar d1 = nonOrthDc[f]*(p1[n] - p1[o]);
    const scalar d2 = nonOrthDc[f]*(p2[n] - p2[o]);

    const scalar magOrth = sqrt(d0*d0 + d1*d1 + d2*d2);
    const scalar magCorr = sqrt(c0*c0 + c1*c1 + c2*c2);
    scalar limiter = (psi * magOrth) / ((1.0 - psi) * magCorr + 1.0e-15);
    if (limiter > 1.0) limiter = 1.0;

    const scalar s = gammaf[f] * magSf[f] * limiter;
    f0[f] = s * c0;
    f1[f] = s * c1;
    f2[f] = s * c2;
}
} // namespace

void deviceLaplacianCorrFluxLimitedVec(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& gammafInt,
    const DeviceBuffer<scalar>& p0,
    const DeviceBuffer<scalar>& p1,
    const DeviceBuffer<scalar>& p2,
    const DeviceBuffer<scalar>* gxc,     // [3]
    const DeviceBuffer<scalar>* gyc,
    const DeviceBuffer<scalar>* gzc,
    scalar psi,
    DeviceBuffer<scalar>* ffc)           // [3], resized here
{
    const int nIf = dm.nInternalFaces;
    for (int k = 0; k < 3; ++k) ffc[k].resize(nIf);
    lapCorrFaceLimitedVecKernel<<<nBlocks(nIf), TPB>>>(
        nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), gammafInt.data(), dm.magSf.data(),
        dm.corrVecX.data(), dm.corrVecY.data(), dm.corrVecZ.data(), dm.nonOrthDc.data(),
        p0.data(), p1.data(), p2.data(),
        gxc[0].data(), gyc[0].data(), gzc[0].data(),
        gxc[1].data(), gyc[1].data(), gzc[1].data(),
        gxc[2].data(), gyc[2].data(), gzc[2].data(),
        psi, ffc[0].data(), ffc[1].data(), ffc[2].data());
    cudaCheck(cudaGetLastError(), "lapCorrFaceLimitedVec");
}


// src = -V*fvc::div(ffc) (the integrated face-flux divergence, = fvm::laplacian.source()'s correction term).
void deviceFaceDivSource(const DeviceMesh& dm, const DeviceBuffer<scalar>& ffc, DeviceBuffer<scalar>& src)
{
    const int nC = dm.nCells;
    src.resize(nC);
    lapCorrGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), ffc.data(), src.data());
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
    divFaceLinearKernel<<<nBlocks(nIf), TPB>>>(nIf, phiInt.data(), dm.w.data(), upper.data(), lower.data());
    cudaCheck(cudaGetLastError(), "divFaceLinear");
    diagGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), upper.data(), lower.data(), diag.data());
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
    divFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, phiInt.data(), upper.data(), lower.data());
    cudaCheck(cudaGetLastError(), "divFace");
    diagGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), upper.data(), lower.data(), diag.data());
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
    divLimitedFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), phiInt.data(), field.data(),
                                                gx.data(), gy.data(), gz.data(),
                                                dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(),
                                                dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
                                                twoByk, upper.data(), lower.data());
    cudaCheck(cudaGetLastError(), "divLimitedFace");
    diagGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), upper.data(), lower.data(), diag.data());
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
    divLimitedVFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), phiInt.data(),
                                                 U[0].data(), U[1].data(), U[2].data(),
                                                 gUx[0].data(), gUy[0].data(), gUz[0].data(),
                                                 gUx[1].data(), gUy[1].data(), gUz[1].data(),
                                                 gUx[2].data(), gUy[2].data(), gUz[2].data(),
                                                 dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(),
                                                 dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
                                                 twoByk, upper.data(), lower.data());
    cudaCheck(cudaGetLastError(), "divLimitedVFace");
    diagGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), upper.data(), lower.data(), diag.data());
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
    divLimitedVFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), phiInt.data(),
                                                 U[0].data(), U[1].data(), U[2].data(),
                                                 g + 0*nC, g + 3*nC, g + 6*nC,     // d(U_0)/d{x,y,z}
                                                 g + 1*nC, g + 4*nC, g + 7*nC,     // d(U_1)/d{x,y,z}
                                                 g + 2*nC, g + 5*nC, g + 8*nC,     // d(U_2)/d{x,y,z}
                                                 dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(),
                                                 dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
                                                 twoByk, upper.data(), lower.data());
    cudaCheck(cudaGetLastError(), "divLimitedVFacePacked");
    diagGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), upper.data(), lower.data(), diag.data());
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
    foldPressureKernel<<<nBlocks(nC), TPB>>>(nC, dm.bndCellStart.data(), dm.bndPerm.data(), rawDiag.data(),
                                             dm.V.data(), divPhi.data(), iC.data(), bC.data(), diagC.data(), b.data());
    cudaCheck(cudaGetLastError(), "foldPressure");
}


namespace {
__global__
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
    foldKernel<<<nBlocks(nC), TPB>>>(nC, dm.bndCellStart.data(), dm.bndPerm.data(), rawDiag.data(),
                                     source.data(), iC.data(), bC.data(), diagC.data(), b.data());
    cudaCheck(cudaGetLastError(), "fold");
}

} // namespace brae
