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

// <limiter><NVDVTVDV> (NVDVTVDV.H:96-132): the V forms' one limiter for all three components, of the selector `lim`
// (limiterOfR). scalarOnV: the control limiterScalarOnV, NVDTVD's r of rhoU_x alone.
__device__ __forceinline__ scalar limitedV(scalar faceFlux, const vector& phiP, const vector& phiN,
                                           const Grads& G, int P, int N, scalar dx, scalar dy, scalar dz,
                                           scalar lim, bool scalarOnV)
{
    if (scalarOnV)
    {
        const scalar gf = phiN.x - phiP.x;
        const scalar gcf = dDotGradJ(G, 0, (faceFlux > 0) ? P : N, dx, dy, dz);
        const scalar rx = (fabs(gcf) >= 1000*fabs(gf)) ? 2*1000*sgnD(gcf)*sgnD(gf) - 1 : 2*(gcf/gf) - 1;
        return limiterOfR(rx, lim);
    }
    const vector gradfV{phiN.x - phiP.x, phiN.y - phiP.y, phiN.z - phiP.z};
    const scalar gradf = gradfV.x*gradfV.x + gradfV.y*gradfV.y + gradfV.z*gradfV.z;
    const int up = (faceFlux > 0) ? P : N;
    const vector dg{dDotGradJ(G, 0, up, dx, dy, dz), dDotGradJ(G, 1, up, dx, dy, dz), dDotGradJ(G, 2, up, dx, dy, dz)};
    const scalar gradcf = gradfV.x*dg.x + gradfV.y*dg.y + gradfV.z*dg.z;
    scalar r;
    if (fabs(gradcf) >= 1000*fabs(gradf)) r = 2*1000*sgnD(gradcf)*sgnD(gradf) - 1;
    else                                 r = 2*(gradcf/gradf) - 1;
    return limiterOfR(r, lim);
}

// limitedSurfaceInterpolationScheme::weights, for the control's limiter 0 and the V limiter
__device__ __forceinline__ scalar limitedW(scalar limiter, scalar cdw, scalar dir)
{
    return limiter*cdw + (1.0 - limiter)*pos0D(dir);
}

// rPsi = 1/psi and c = sqrt(thermo.Cp()/thermo.Cv()*rPsi) (rhoCentralFoam.C:107, 135). hConst: Cp/Cv is one number;
// janaf (milestone I): Cp(T)/Cv(T) at each cell's or face's own T, the fields thermo.Cp() and thermo.Cv() hold
__global__ void auxKernel(int n, scalar CpByCv, const scalar* __restrict__ psi, scalar* __restrict__ rPsi,
                          scalar* __restrict__ c, GasCoeffs gas, const scalar* __restrict__ T)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    rPsi[i] = 1.0/psi[i];
    if (T)
    {
        const scalar r = gasCp(scalar(0), T[i], gas)/gasCv(scalar(0), T[i], gas);
        c[i] = sqrt(r*rPsi[i]);
        return;
    }
    c[i] = sqrt(CpByCv*rPsi[i]);
}

struct CellInputs
{
    const scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rPsi, *e, *c;
};
struct FaceOutputs
{
    scalar *phi, *phiUx, *phiUy, *phiUz, *phiEp, *amaxSf;
    scalar *aUx = nullptr, *aUy = nullptr, *aUz = nullptr;   // viscous runs only
};

__device__ __forceinline__ void store(const FaceFlux& o, const FaceOutputs& O, int i)
{
    O.phi[i] = o.phi;
    O.phiUx[i] = o.phiUx;
    O.phiUy[i] = o.phiUy;
    O.phiUz[i] = o.phiUz;
    O.phiEp[i] = o.phiEp;
    O.amaxSf[i] = o.amaxSf;
    if (O.aUx)
    {
        O.aUx[i] = o.aUx;
        O.aUy[i] = o.aUy;
        O.aUz[i] = o.aUz;
    }
}

__global__ void faceKernel(int nIF, bool tadmor, int ctrl,
                           const label* __restrict__ own, const label* __restrict__ nei,
                           const scalar* __restrict__ w,
                           const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                           const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                           const scalar* __restrict__ Cx, const scalar* __restrict__ Cy,
                           const scalar* __restrict__ Cz,
                           scalar limRho, scalar limT, scalar limU,
                           CellInputs F, Grads G, FaceOutputs O, const scalar* __restrict__ meshPhi)
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
        auto scalarRecon = [&](const scalar* q, int gi, scalar lim)
        {
            const scalar wq = (ctrl == controlUpwind)
                ? limitedW(0.0, cdw, dir)
                : limitedFaceWeight(P, N, dir, cdw, q, G.x[gi], G.y[gi], G.z[gi], d, lim);
            return wq*(q[P] - q[N]) + q[N];
        };
        in.rho[k] = scalarRecon(F.rho, gRho, limRho);
        const scalar wU = limitedW((ctrl == controlUpwind)
                                       ? 0.0
                                       : limitedV(dir, UP, UN, G, P, N, dx, dy, dz, limU,
                                                  ctrl == controlLimiterScalarOnV),
                                   cdw, dir);
        in.rhoU[k] = vector{wU*(UP.x - UN.x) + UN.x, wU*(UP.y - UN.y) + UN.y, wU*(UP.z - UN.z) + UN.z};
        in.rPsi[k] = scalarRecon(F.rPsi, gRPsi, limT);
        in.e[k] = scalarRecon(F.e, gE, limT);
        in.c[k] = scalarRecon(F.c, gC, limT);
    }
    store(centralFaceFlux(in, Sfx[f], Sfy[f], Sfz[f], magSf[f], tadmor, ctrl, meshPhi != nullptr,
                          meshPhi ? meshPhi[f] : scalar(0)), O, f);
}

__global__ void patchFaceKernel(int nB, int nIF, bool tadmor, int ctrl, scalar deltaT,
                                const label* __restrict__ isEmpty,
                                const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                                const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                                CellInputs B, FaceOutputs O, double* __restrict__ outflow,
                                const scalar* __restrict__ meshPhiB)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    if (isEmpty[k])
    {
        O.phi[k] = O.phiUx[k] = O.phiUy[k] = O.phiUz[k] = O.phiEp[k] = O.amaxSf[k] = 0;
        if (O.aUx) O.aUx[k] = O.aUy[k] = O.aUz[k] = 0;
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
    const FaceFlux o = centralFaceFlux(in, Sfx[f], Sfy[f], Sfz[f], magSf[f], tadmor, ctrl, meshPhiB != nullptr,
                                       meshPhiB ? meshPhiB[k] : scalar(0));
    store(o, O, k);
    // this step's sums (outflow[5..9]); outflowStepKernel adds them, times deltaT, to the run's totals as the host
    // arm does. Adding every face straight into the run's total rounds each of ~1e7 additions at the TOTAL's scale:
    // on the cone that was 7.6e-15 of y-momentum, the diagnostic's own error, against the host arm's 3e-17.
    atomicAdd(&outflow[5], o.phi);
    atomicAdd(&outflow[6], o.phiUx);
    atomicAdd(&outflow[7], o.phiUy);
    atomicAdd(&outflow[8], o.phiUz);
    atomicAdd(&outflow[9], o.phiEp);
}

__global__ void outflowStepKernel(double* __restrict__ outflow, scalar deltaT)
{
    const int i = threadIdx.x;
    if (i >= 5) return;
    outflow[i] += deltaT*outflow[5 + i];
    outflow[5 + i] = 0;
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
                   double* outflow, scalar deltaT, const scalar* meshPhiI, const scalar* meshPhiB)
{
    const int nC = cs.nC, nB = cs.nB, nIF = cs.nIF;
    const scalar CpByCv = gasCp(cs.gas)/gasCv(cs.gas);
    // the limiters (Case: reconstruct(rho), (T), (U)); the control limiterVanLeer replaces them all by vanLeer
    const bool allVanLeer = (control == controlLimiterVanLeer);
    const scalar limRho = allVanLeer ? kVanLeerTwoByk : cs.limRho;
    const scalar limT = allVanLeer ? kVanLeerTwoByk : cs.limT;
    const scalar limU = allVanLeer ? kVanLeerTwoByk : cs.limU;

    const bool janaf = cs.gas.janaf != 0;
    auxKernel<<<blocksFor(nC), kTPB>>>(nC, CpByCv, F.psi.data(), X.rPsi.data(), X.c.data(), cs.gas,
                                       janaf ? F.T.data() : nullptr);
    if (nB) auxKernel<<<blocksFor(nB), kTPB>>>(nB, CpByCv, F.psiB.data(), X.rPsiB.data(), X.cB.data(), cs.gas,
                                               janaf ? F.TB.data() : nullptr);
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
        FaceOutputs O{X.phi.data(), X.phiUx.data(), X.phiUy.data(), X.phiUz.data(), X.phiEp.data(),
                      X.amaxSf.data()};
        if (cs.viscous)
        {
            X.aUx.resize(nIF);
            X.aUy.resize(nIF);
            X.aUz.resize(nIF);
            O.aUx = X.aUx.data();
            O.aUy = X.aUy.data();
            O.aUz = X.aUz.data();
        }
        faceKernel<<<blocksFor(nIF), kTPB>>>(nIF, cs.tadmor, control, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                             dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), dm.magSf.data(),
                                             dm.Cx.data(), dm.Cy.data(), dm.Cz.data(), limRho, limT, limU, Fi, G, O,
                                             meshPhiI);
        cudaCheck(cudaGetLastError(), "rhoCentral faces");
    }
    if (nB)
    {
        const CellInputs Bi{F.rhoB.data(), F.rhoUxB.data(), F.rhoUyB.data(), F.rhoUzB.data(), X.rPsiB.data(),
                            F.eB.data(), X.cB.data()};
        FaceOutputs O{X.phiB.data(), X.phiUxB.data(), X.phiUyB.data(), X.phiUzB.data(), X.phiEpB.data(),
                            X.amaxSfB.data()};
        if (cs.viscous)
        {
            X.aUxB.resize(nB);
            X.aUyB.resize(nB);
            X.aUzB.resize(nB);
            O.aUx = X.aUxB.data();
            O.aUy = X.aUyB.data();
            O.aUz = X.aUzB.data();
        }
        patchFaceKernel<<<blocksFor(nB), kTPB>>>(nB, nIF, cs.tadmor, control, deltaT, dm.bndIsEmpty.data(),
                                                 dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), dm.magSf.data(), Bi, O,
                                                 outflow, meshPhiB);
        cudaCheck(cudaGetLastError(), "rhoCentral patch faces");
        outflowStepKernel<<<1, 32>>>(outflow, deltaT);
        cudaCheck(cudaGetLastError(), "rhoCentral outflow");
    }
}

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
