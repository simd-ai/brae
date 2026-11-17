// rhoCentralFlux.cu -- the central-upwind flux assembly on the device. See the header.
//
// Shared infrastructure it runs on:
//   deviceGaussGradFused (device_fvc.cu)   the limiters' gradients, three fields a launch, OpenFOAM's face order
//   limitedFaceWeight + limiterOfR          the scalar vanLeer reconstruction (NVDTVD), device_limiter.cuh
//   centralFaceFlux                         the per-face flux both arms evaluate (rhoCentralFaceFlux.cuh)
// The vector vanLeerV limiter (NVDVTVDV) is here: brae's other copy is inlined in divLimitedVFaceKernel.
#include "rhoCentralFlux.cuh"
#include "rhoCentralFaceFlux.cuh"
#include "rhoCentralThermo.cuh"

#include "device_mesh.cuh"      // kVanLeerTwoByk, before device_limiter.cuh
#include "device_limiter.cuh"

#include <cuda_runtime.h>

namespace brae {
namespace gpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

namespace {

constexpr int kTPB = 256;
inline int blocksFor(int n) { return (n + kTPB - 1)/kTPB; }

__device__ __forceinline__ scalar sgnD(scalar s) { return (s >= 0) ? 1.0 : -1.0; }    // Foam::sign
__device__ __forceinline__ scalar pos0D(scalar s) { return (s >= 0) ? 1.0 : 0.0; }   // Foam::pos0

struct Grads   // the seven component gradients, SoA
{
    const scalar* x[7];
    const scalar* y[7];
    const scalar* z[7];
};
enum { gRho = 0, gUx = 1, gUy = 2, gUz = 3, gRPsi = 4, gE = 5, gC = 6 };

// (d & gradc)_j for the tensor held as component gradients j = 0..2 (TensorI.H inner product)
__device__ __forceinline__ scalar dDotGradJ(const Grads& G, int j, int cell, scalar dx, scalar dy, scalar dz)
{
    return dx*G.x[gUx + j][cell] + dy*G.y[gUx + j][cell] + dz*G.z[gUx + j][cell];
}

// vanLeerLimiter<NVDVTVDV> (NVDVTVDV.H:96-132)
__device__ __forceinline__ scalar vanLeerV(scalar faceFlux, const vector& phiP, const vector& phiN,
                                           const Grads& G, int P, int N, scalar dx, scalar dy, scalar dz)
{
    const vector gradfV{phiN.x - phiP.x, phiN.y - phiP.y, phiN.z - phiP.z};
    const scalar gradf = gradfV.x*gradfV.x + gradfV.y*gradfV.y + gradfV.z*gradfV.z;
    const int up = (faceFlux > 0) ? P : N;
    const vector dg{dDotGradJ(G, 0, up, dx, dy, dz), dDotGradJ(G, 1, up, dx, dy, dz), dDotGradJ(G, 2, up, dx, dy, dz)};
    const scalar gradcf = gradfV.x*dg.x + gradfV.y*dg.y + gradfV.z*dg.z;
    scalar r;
    if (fabs(gradcf) >= 1000*fabs(gradf)) r = 2*1000*sgnD(gradcf)*sgnD(gradf) - 1;
    else                                 r = 2*(gradcf/gradf) - 1;
    return (r + fabs(r))/(1 + fabs(r));
}

// limitedSurfaceInterpolationScheme::weights, for the control's limiter 0 and the V limiter
__device__ __forceinline__ scalar limitedW(scalar limiter, scalar cdw, scalar dir)
{
    return limiter*cdw + (1.0 - limiter)*pos0D(dir);
}

__global__ void auxKernel(int n, scalar CpByCv, const scalar* __restrict__ psi, scalar* __restrict__ rPsi,
                          scalar* __restrict__ c)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    rPsi[i] = 1.0/psi[i];
    c[i] = sqrt(CpByCv*rPsi[i]);
}

struct CellInputs
{
    const scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rPsi, *e, *c;
};
struct FaceOutputs
{
    scalar *phi, *phiUx, *phiUy, *phiUz, *phiEp, *amaxSf;
};

__device__ __forceinline__ void store(const FaceFlux& o, const FaceOutputs& O, int i)
{
    O.phi[i] = o.phi;
    O.phiUx[i] = o.phiUx;
    O.phiUy[i] = o.phiUy;
    O.phiUz[i] = o.phiUz;
    O.phiEp[i] = o.phiEp;
    O.amaxSf[i] = o.amaxSf;
}

__global__ void faceKernel(int nIF, bool tadmor, int ctrl,
                           const label* __restrict__ own, const label* __restrict__ nei,
                           const scalar* __restrict__ w,
                           const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                           const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                           const scalar* __restrict__ Cx, const scalar* __restrict__ Cy,
                           const scalar* __restrict__ Cz,
                           CellInputs F, Grads G, FaceOutputs O)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const int P = own[f], N = nei[f];
    const scalar dx = Cx[N] - Cx[P], dy = Cy[N] - Cy[P], dz = Cz[N] - Cz[P];
    const vector d{dx, dy, dz};
    const scalar cdw = w[f];
    FaceStates in;
    const vector UP{F.rhoUx[P], F.rhoUy[P], F.rhoUz[P]};
    const vector UN{F.rhoUx[N], F.rhoUy[N], F.rhoUz[N]};
#pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        const scalar dir = (k == 0) ? 1.0 : -1.0;   // the `pos` and `neg` surface fields, createFields.H
        // interpolate(q, dir, "reconstruct(...)"): vanLeer weights, value = w*(P - N) + N (dotInterpolate's form)
        auto scalarRecon = [&](const scalar* q, int gi)
        {
            const scalar wq = (ctrl == controlUpwind)
                ? limitedW(0.0, cdw, dir)
                : limitedFaceWeight(P, N, dir, cdw, q, G.x[gi], G.y[gi], G.z[gi], d, kVanLeerTwoByk);
            return wq*(q[P] - q[N]) + q[N];
        };
        in.rho[k] = scalarRecon(F.rho, gRho);
        const scalar wU = limitedW((ctrl == controlUpwind) ? 0.0 : vanLeerV(dir, UP, UN, G, P, N, dx, dy, dz),
                                   cdw, dir);
        in.rhoU[k] = vector{wU*(UP.x - UN.x) + UN.x, wU*(UP.y - UN.y) + UN.y, wU*(UP.z - UN.z) + UN.z};
        in.rPsi[k] = scalarRecon(F.rPsi, gRPsi);
        in.e[k] = scalarRecon(F.e, gE);
        in.c[k] = scalarRecon(F.c, gC);
    }
    store(centralFaceFlux(in, Sfx[f], Sfy[f], Sfz[f], magSf[f], tadmor, ctrl), O, f);
}

__global__ void patchFaceKernel(int nB, int nIF, bool tadmor, int ctrl, scalar deltaT,
                                const label* __restrict__ isEmpty,
                                const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                                const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                                CellInputs B, FaceOutputs O, double* __restrict__ outflow)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    if (isEmpty[k])
    {
        O.phi[k] = O.phiUx[k] = O.phiUy[k] = O.phiUz[k] = O.phiEp[k] = O.amaxSf[k] = 0;
        return;
    }
    FaceStates in;
#pragma unroll
    for (int j = 0; j < 2; ++j)
    {
        in.rho[j] = B.rho[k];
        in.rhoU[j] = vector{B.rhoUx[k], B.rhoUy[k], B.rhoUz[k]};
        in.rPsi[j] = B.rPsi[k];
        in.e[j] = B.e[k];
        in.c[j] = B.c[k];
    }
    const int f = nIF + k;
    const FaceFlux o = centralFaceFlux(in, Sfx[f], Sfy[f], Sfz[f], magSf[f], tadmor, ctrl);
    store(o, O, k);
    atomicAdd(&outflow[0], deltaT*o.phi);
    atomicAdd(&outflow[1], deltaT*o.phiUx);
    atomicAdd(&outflow[2], deltaT*o.phiUy);
    atomicAdd(&outflow[3], deltaT*o.phiUz);
    atomicAdd(&outflow[4], deltaT*o.phiEp);
}

} // namespace


void DeviceFluxes::allocate(int nC, int nIF, int nB)
{
    rPsi.resize(nC);
    c.resize(nC);
    rPsiB.resize(nB);
    cB.resize(nB);
    for (int i = 0; i < 7; ++i)
    {
        gx[i].resize(nC);
        gy[i].resize(nC);
        gz[i].resize(nC);
    }
    for (DeviceBuffer<scalar>* b : {&phi, &phiUx, &phiUy, &phiUz, &phiEp, &amaxSf}) b->resize(nIF);
    for (DeviceBuffer<scalar>* b : {&phiB, &phiUxB, &phiUyB, &phiUzB, &phiEpB, &amaxSfB}) b->resize(nB);
}


void centralFluxes(const DeviceMesh& dm, const Case& cs, int control, DeviceFields& F, DeviceFluxes& X,
                   double* outflow, scalar deltaT)
{
    const int nC = cs.nC, nB = cs.nB, nIF = cs.nIF;
    const scalar CpByCv = gasCp(cs.gas)/gasCv(cs.gas);

    auxKernel<<<blocksFor(nC), kTPB>>>(nC, CpByCv, F.psi.data(), X.rPsi.data(), X.c.data());
    if (nB) auxKernel<<<blocksFor(nB), kTPB>>>(nB, CpByCv, F.psiB.data(), X.rPsiB.data(), X.cB.data());
    cudaCheck(cudaGetLastError(), "rhoCentral aux");

    // the limiters' gradients, three fields a launch (deviceGaussGradFused, bit-identical to one at a time)
    {
        const DeviceBuffer<scalar>* vol[7] = {&F.rho, &F.rhoUx, &F.rhoUy, &F.rhoUz, &X.rPsi, &F.e, &X.c};
        const DeviceBuffer<scalar>* bv[7] = {&F.rhoB, &F.rhoUxB, &F.rhoUyB, &F.rhoUzB, &X.rPsiB, &F.eB, &X.cB};
        deviceGaussGradFused(dm, 3, vol, bv, X.gx, X.gy, X.gz, nullptr);
        deviceGaussGradFused(dm, 3, vol + 3, bv + 3, X.gx + 3, X.gy + 3, X.gz + 3, nullptr);
        deviceGaussGradFused(dm, 1, vol + 6, bv + 6, X.gx + 6, X.gy + 6, X.gz + 6, nullptr);
    }
    Grads G;
    for (int i = 0; i < 7; ++i)
    {
        G.x[i] = X.gx[i].data();
        G.y[i] = X.gy[i].data();
        G.z[i] = X.gz[i].data();
    }
    if (nIF)
    {
        const CellInputs Fi{F.rho.data(), F.rhoUx.data(), F.rhoUy.data(), F.rhoUz.data(), X.rPsi.data(), F.e.data(),
                            X.c.data()};
        const FaceOutputs O{X.phi.data(), X.phiUx.data(), X.phiUy.data(), X.phiUz.data(), X.phiEp.data(),
                            X.amaxSf.data()};
        faceKernel<<<blocksFor(nIF), kTPB>>>(nIF, cs.tadmor, control, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                             dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), dm.magSf.data(),
                                             dm.Cx.data(), dm.Cy.data(), dm.Cz.data(), Fi, G, O);
        cudaCheck(cudaGetLastError(), "rhoCentral faces");
    }
    if (nB)
    {
        const CellInputs Bi{F.rhoB.data(), F.rhoUxB.data(), F.rhoUyB.data(), F.rhoUzB.data(), X.rPsiB.data(),
                            F.eB.data(), X.cB.data()};
        const FaceOutputs O{X.phiB.data(), X.phiUxB.data(), X.phiUyB.data(), X.phiUzB.data(), X.phiEpB.data(),
                            X.amaxSfB.data()};
        patchFaceKernel<<<blocksFor(nB), kTPB>>>(nB, nIF, cs.tadmor, control, deltaT, dm.bndIsEmpty.data(),
                                                 dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), dm.magSf.data(), Bi, O,
                                                 outflow);
        cudaCheck(cudaGetLastError(), "rhoCentral patch faces");
    }
}

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
