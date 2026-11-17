// rhoCentralFoam_device.cu -- the CUDA arm. See the header; the arithmetic is rhoCentralFoam_cpp.cu's.
#include "rhoCentralFoam_device.cuh"
#include "rhoCentralThermo.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace rhoCentral {

namespace {

constexpr int kTPB = 256;
inline int blocksFor(int n) { return (n + kTPB - 1)/kTPB; }

__device__ __forceinline__ scalar sgnD(scalar s) { return (s >= 0) ? 1.0 : -1.0; }       // Foam::sign
__device__ __forceinline__ scalar pos0D(scalar s) { return (s >= 0) ? 1.0 : 0.0; }      // Foam::pos0
__device__ __forceinline__ scalar maxOF(scalar a, scalar b) { return (a > b) ? a : b; } // Foam::max
__device__ __forceinline__ scalar minOF(scalar a, scalar b) { return (a < b) ? a : b; } // Foam::min

// vanLeerLimiter<NVDTVD> (vanLeer.H, NVDTVD.H:96-121); gP, gN are the upwind candidates' gradients
__device__ __forceinline__ scalar vanLeerS(scalar faceFlux, scalar phiP, scalar phiN,
                                           scalar gPx, scalar gPy, scalar gPz,
                                           scalar gNx, scalar gNy, scalar gNz,
                                           scalar dx, scalar dy, scalar dz)
{
    const scalar gradf = phiN - phiP;
    const scalar gradcf = (faceFlux > 0) ? (dx*gPx + dy*gPy + dz*gPz) : (dx*gNx + dy*gNy + dz*gNz);
    scalar r;
    if (fabs(gradcf) >= 1000*fabs(gradf)) r = 2*1000*sgnD(gradcf)*sgnD(gradf) - 1;
    else                                 r = 2*(gradcf/gradf) - 1;
    return (r + fabs(r))/(1 + fabs(r));
}

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

__device__ __forceinline__ scalar limitedW(scalar limiter, scalar cdw, scalar dir)
{
    return limiter*cdw + (1.0 - limiter)*pos0D(dir);
}

struct FaceVals
{
    scalar rho[2], rPsi[2], e[2], c[2];
    vector rhoU[2];
};

// rhoCentralFoam.C:114-210 for one face; the same expressions as rhoCentralFoam_cpp.cu faceFlux
__device__ __forceinline__ void centralFlux(const FaceVals& in, scalar Sx, scalar Sy, scalar Sz, scalar magSf,
                                            bool tadmor, int ctrl, scalar& phi, scalar& phiUx, scalar& phiUy, scalar& phiUz,
                                            scalar& phiEp, scalar& amaxSfOut)
{
    const vector U_pos{in.rhoU[0].x/in.rho[0], in.rhoU[0].y/in.rho[0], in.rhoU[0].z/in.rho[0]};
    const vector U_neg{in.rhoU[1].x/in.rho[1], in.rhoU[1].y/in.rho[1], in.rhoU[1].z/in.rho[1]};
    const scalar p_pos = in.rho[0]*in.rPsi[0];
    const scalar p_neg = in.rho[1]*in.rPsi[1];
    scalar phiv_pos = U_pos.x*Sx + U_pos.y*Sy + U_pos.z*Sz;
    scalar phiv_neg = U_neg.x*Sx + U_neg.y*Sy + U_neg.z*Sz;
    const scalar cSf_pos = in.c[0]*magSf;
    const scalar cSf_neg = in.c[1]*magSf;
    const scalar ap = maxOF(maxOF(phiv_pos + cSf_pos, phiv_neg + cSf_neg), 0.0);
    const scalar am = minOF(minOF(phiv_pos - cSf_pos, phiv_neg - cSf_neg), 0.0);
    scalar a_pos = ap/(ap - am);
    scalar amaxSf = maxOF(fabs(am), fabs(ap));
    scalar aSf = am*a_pos;
    if (tadmor)
    {
        aSf = -0.5*amaxSf;
        a_pos = 0.5;
    }
    const scalar a_neg = 1.0 - a_pos;
    phiv_pos *= a_pos;
    phiv_neg *= a_neg;
    const scalar aphiv_pos = phiv_pos - aSf;
    const scalar aphiv_neg = phiv_neg + aSf;
    amaxSf = maxOF(fabs(aphiv_pos), fabs(aphiv_neg));

    amaxSfOut = amaxSf;
    phi = aphiv_pos*in.rho[0] + aphiv_neg*in.rho[1];
    const scalar pf = a_pos*p_pos + a_neg*p_neg;
    phiUx = (aphiv_pos*in.rhoU[0].x + aphiv_neg*in.rhoU[1].x) + pf*Sx;
    phiUy = (aphiv_pos*in.rhoU[0].y + aphiv_neg*in.rhoU[1].y) + pf*Sy;
    phiUz = (aphiv_pos*in.rhoU[0].z + aphiv_neg*in.rhoU[1].z) + pf*Sz;
    const scalar K_pos = U_pos.x*U_pos.x + U_pos.y*U_pos.y + U_pos.z*U_pos.z;
    const scalar K_neg = U_neg.x*U_neg.x + U_neg.y*U_neg.y + U_neg.z*U_neg.z;
    phiEp = aphiv_pos*(in.rho[0]*(in.e[0] + 0.5*K_pos) + p_pos)
          + aphiv_neg*(in.rho[1]*(in.e[1] + 0.5*K_neg) + p_neg);
    // the pressure-jump term of the energy flux, which the CONTROL drops (see Control in the header)
    if (ctrl != controlNoPressureJump) phiEp = phiEp + aSf*p_pos - aSf*p_neg;
}

__global__ void auxKernel(int n, scalar CpByCv, const scalar* __restrict__ psi, scalar* __restrict__ rPsi,
                          scalar* __restrict__ c)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    rPsi[i] = 1.0/psi[i];
    c[i] = sqrt(CpByCv*rPsi[i]);
}

struct CellFields
{
    const scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rPsi, *e, *c;
};

__global__ void faceKernel(int nIF, bool tadmor, int ctrl,
                           const label* __restrict__ own, const label* __restrict__ nei,
                           const scalar* __restrict__ w,
                           const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                           const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                           const scalar* __restrict__ Cx, const scalar* __restrict__ Cy,
                           const scalar* __restrict__ Cz,
                           CellFields F, Grads G,
                           scalar* __restrict__ phi, scalar* __restrict__ phiUx, scalar* __restrict__ phiUy,
                           scalar* __restrict__ phiUz, scalar* __restrict__ phiEp, scalar* __restrict__ amaxSf)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const int P = own[f], N = nei[f];
    const scalar dx = Cx[N] - Cx[P], dy = Cy[N] - Cy[P], dz = Cz[N] - Cz[P];
    const scalar cdw = w[f];
    FaceVals in;
    const vector UP{F.rhoUx[P], F.rhoUy[P], F.rhoUz[P]};
    const vector UN{F.rhoUx[N], F.rhoUy[N], F.rhoUz[N]};
#pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        const scalar dir = (k == 0) ? 1.0 : -1.0;
        auto scalarRecon = [&](const scalar* q, int gi)
        {
            const scalar lim = (ctrl == controlUpwind) ? 0.0
                             : vanLeerS(dir, q[P], q[N], G.x[gi][P], G.y[gi][P], G.z[gi][P],
                                        G.x[gi][N], G.y[gi][N], G.z[gi][N], dx, dy, dz);
            const scalar wq = limitedW(lim, cdw, dir);
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
    centralFlux(in, Sfx[f], Sfy[f], Sfz[f], magSf[f], tadmor, ctrl, phi[f], phiUx[f], phiUy[f], phiUz[f], phiEp[f],
                amaxSf[f]);
}

struct BndFields
{
    const scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rPsi, *e, *c;
};

__global__ void bndFaceKernel(int nB, int nIF, bool tadmor, int ctrl, scalar deltaT,
                              const label* __restrict__ isEmpty,
                              const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                              const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                              BndFields B,
                              scalar* __restrict__ phi, scalar* __restrict__ phiUx, scalar* __restrict__ phiUy,
                              scalar* __restrict__ phiUz, scalar* __restrict__ phiEp, scalar* __restrict__ amaxSf,
                              double* __restrict__ running)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    if (isEmpty[k])
    {
        phi[k] = phiUx[k] = phiUy[k] = phiUz[k] = phiEp[k] = amaxSf[k] = 0;
        return;
    }
    FaceVals in;
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
    centralFlux(in, Sfx[f], Sfy[f], Sfz[f], magSf[f], tadmor, ctrl, phi[k], phiUx[k], phiUy[k], phiUz[k], phiEp[k],
                amaxSf[k]);
    atomicAdd(&running[0], deltaT*phi[k]);
    atomicAdd(&running[1], deltaT*phiUx[k]);
    atomicAdd(&running[2], deltaT*phiUy[k]);
    atomicAdd(&running[3], deltaT*phiUz[k]);
    atomicAdd(&running[4], deltaT*phiEp[k]);
}

struct CellState
{
    scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rhoE, *e, *T, *Told, *p, *psi, *Ux, *Uy, *Uz;
};
struct Fluxes
{
    const scalar *phi, *phiUx, *phiUy, *phiUz, *phiEp, *amaxSf;
    const scalar *phiB, *phiUxB, *phiUyB, *phiUzB, *phiEpB, *amaxSfB;
};

// surfaceIntegrate's order: owner (+) and neighbour (-) faces merged by face index, then non-empty patch faces;
// then / V (fvcSurfaceIntegrate.C:45-78). The diagonal solves, U, e, thermo.correct()'s cell half and p follow.
__global__ void cellKernel(int nC, int slot, scalar rDeltaT, int v0, int v1, int v2, GasCoeffs gas,
                           const label* __restrict__ ownerStart, const label* __restrict__ losort,
                           const label* __restrict__ losortStart, const label* __restrict__ bndCellStart,
                           const label* __restrict__ bndPerm, const label* __restrict__ bndIsEmpty,
                           const scalar* __restrict__ V, Fluxes Fx, CellState S,
                           scalar* __restrict__ coCell, int* __restrict__ status)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, sa = 0;
    int fo = ownerStart[c];
    const int foEnd = ownerStart[c + 1];
    int kn = losortStart[c];
    const int knEnd = losortStart[c + 1];
    while (fo < foEnd || kn < knEnd)
    {
        if ((kn >= knEnd) || (fo < foEnd && fo < losort[kn]))
        {
            d0 += Fx.phi[fo];
            d1 += Fx.phiUx[fo];
            d2 += Fx.phiUy[fo];
            d3 += Fx.phiUz[fo];
            d4 += Fx.phiEp[fo];
            sa += Fx.amaxSf[fo];
            ++fo;
        }
        else
        {
            const int f = losort[kn];
            d0 -= Fx.phi[f];
            d1 -= Fx.phiUx[f];
            d2 -= Fx.phiUy[f];
            d3 -= Fx.phiUz[f];
            d4 -= Fx.phiEp[f];
            sa += Fx.amaxSf[f];   // surfaceSum: += on both sides (fvcSurfaceIntegrate.C:165-166)
            ++kn;
        }
    }
    for (int q = bndCellStart[c]; q < bndCellStart[c + 1]; ++q)
    {
        const int k = bndPerm[q];
        if (bndIsEmpty[k]) continue;
        d0 += Fx.phiB[k];
        d1 += Fx.phiUxB[k];
        d2 += Fx.phiUyB[k];
        d3 += Fx.phiUzB[k];
        d4 += Fx.phiEpB[k];
        sa += Fx.amaxSfB[k];
    }
    const scalar Vc = V[c];
    coCell[c] = sa;
    d0 /= Vc;
    d1 /= Vc;
    d2 /= Vc;
    d3 /= Vc;
    d4 /= Vc;

    const scalar diag = rDeltaT*Vc;
    const scalar rho = (rDeltaT*S.rho[c]*Vc - Vc*d0)/diag;
    scalar rUx = S.rhoUx[c], rUy = S.rhoUy[c], rUz = S.rhoUz[c];
    if (v0) rUx = (rDeltaT*rUx*Vc - Vc*d1)/diag;
    if (v1) rUy = (rDeltaT*rUy*Vc - Vc*d2)/diag;
    if (v2) rUz = (rDeltaT*rUz*Vc - Vc*d3)/diag;
    const scalar Ux = rUx/rho, Uy = rUy/rho, Uz = rUz/rho;
    const scalar rhoE = (rDeltaT*S.rhoE[c]*Vc - Vc*d4)/diag;
    const scalar e = rhoE/rho - 0.5*(Ux*Ux + Uy*Uy + Uz*Uz);
    const scalar Told = S.T[c];
    const scalar pOld = S.p[c];
    const GasTResult r = gasTEs(e, pOld, Told, gas);
    const scalar psi = gasPsi(pOld, r.T, gas);
    const scalar p = rho/psi;

    S.rho[c] = rho;
    S.rhoUx[c] = rUx;
    S.rhoUy[c] = rUy;
    S.rhoUz[c] = rUz;
    S.Ux[c] = Ux;
    S.Uy[c] = Uy;
    S.Uz[c] = Uz;
    S.rhoE[c] = rhoE;
    S.e[c] = e;
    S.Told[c] = Told;
    S.T[c] = r.T;
    S.psi[c] = psi;
    S.p[c] = p;
    // what OpenFOAM would stop on (the inversion), and what makes the state unphysical
    int st = r.status;
    if (!(rho > 0) || !(p > 0) || !(r.T > 0) || !isfinite(e)) st |= 4;
    if (st) atomicOr(&status[slot], st);
}

struct BndState
{
    scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rhoE, *e, *T, *p, *psi, *Ux, *Uy, *Uz;
};
struct CellRead
{
    const scalar *Ux, *Uy, *Uz, *e, *Told, *p;
};

__global__ void bndKernel(int nB, int slot, GasCoeffs gas, const label* __restrict__ bCell,
                          const int* __restrict__ kU, const int* __restrict__ kP, const int* __restrict__ kT,
                          CellRead C, BndState B, int* __restrict__ status)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    const int c = bCell[k];
    const scalar rhoOld = B.rho[k];   // rho's calculated patches are reset only at the end of the step
    const scalar pOld = B.p[k];
    if (kU[k] != bcFixedValue)
    {
        B.Ux[k] = C.Ux[c];
        B.Uy[k] = C.Uy[c];
        B.Uz[k] = C.Uz[c];
    }
    const scalar Ux = B.Ux[k], Uy = B.Uy[k], Uz = B.Uz[k];
    B.rhoUx[k] = rhoOld*Ux;
    B.rhoUy[k] = rhoOld*Uy;
    B.rhoUz[k] = rhoOld*Uz;
    scalar e, T;
    if (kT[k] == bcFixedValue)
    {
        T = B.T[k];
        e = gasEs(pOld, T, gas);
    }
    else
    {
        e = C.e[c];
        const GasTResult r = gasTEs(e, pOld, C.Told[c], gas);   // Tw.evaluate() set T_b to the cell's old T
        if (r.status) atomicOr(&status[slot], r.status);
        T = r.T;
    }
    const scalar psi = gasPsi(pOld, T, gas);
    B.e[k] = e;
    B.T[k] = T;
    B.psi[k] = psi;
    B.rhoE[k] = rhoOld*(e + 0.5*(Ux*Ux + Uy*Uy + Uz*Uz));
    const scalar p = (kP[k] != bcFixedValue) ? C.p[c] : pOld;
    B.p[k] = p;
    B.rho[k] = psi*p;
}

// centralCourantNo.H: CoNum = 0.5*gMax(sumAmaxSf/V)*dt, meanCoNum = 0.5*(gSum(sumAmaxSf)/gSum(V))*dt
__global__ void courantKernel(int nC, const scalar* __restrict__ sumA, const scalar* __restrict__ V,
                              double* __restrict__ out)
{
    __shared__ double smax[1024], ssum[1024], sv[1024];
    const int t = threadIdx.x;
    double mx = -1e300, sa = 0, svol = 0;
    for (int i = t; i < nC; i += blockDim.x)
    {
        const double r = sumA[i]/V[i];
        mx = (r > mx) ? r : mx;
        sa += sumA[i];
        svol += V[i];
    }
    smax[t] = mx;
    ssum[t] = sa;
    sv[t] = svol;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            smax[t] = (smax[t + s] > smax[t]) ? smax[t + s] : smax[t];
            ssum[t] += ssum[t + s];
            sv[t] += sv[t + s];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        out[0] = smax[0];
        out[1] = ssum[0];
        out[2] = sv[0];
    }
}

} // namespace


struct DeviceSolver::Impl
{
    DeviceMesh dm;
    // cells
    DeviceBuffer<scalar> rho, rhoUx, rhoUy, rhoUz, rhoE, e, T, Told, p, psi, Ux, Uy, Uz, rPsi, c, coCell;
    // flat boundary
    DeviceBuffer<scalar> rhoB, rhoUxB, rhoUyB, rhoUzB, rhoEB, eB, TB, pB, psiB, UxB, UyB, UzB, rPsiB, cB;
    DeviceBuffer<label> bCell;
    DeviceBuffer<int> kU, kP, kT;
    DeviceBuffer<scalar> gx[7], gy[7], gz[7];
    DeviceBuffer<scalar> phi, phiUx, phiUy, phiUz, phiEp, amaxSf;
    DeviceBuffer<scalar> phiB, phiUxB, phiUyB, phiUzB, phiEpB, amaxSfB;
    DeviceBuffer<int> status;
    DeviceBuffer<double> running, courant;
};

namespace {
void up(DeviceBuffer<scalar>& d, const std::vector<scalar>& h) { d.copyFrom(h); }
void upCmpt(DeviceBuffer<scalar>& d, const std::vector<vector>& h, int j)
{
    std::vector<scalar> t(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) t[i] = (j == 0) ? h[i].x : (j == 1) ? h[i].y : h[i].z;
    d.copyFrom(t);
}
void downCmpt(const DeviceBuffer<scalar>& d, std::vector<vector>& h, int j)
{
    const std::vector<scalar> t = d.host();
    h.resize(t.size());
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        if (j == 0) h[i].x = t[i];
        else if (j == 1) h[i].y = t[i];
        else h[i].z = t[i];
    }
}
}


DeviceSolver::DeviceSolver(const Case& cs, const State& s0)
  : d_(std::make_unique<Impl>()), cs_(cs)
{
    size_t free0 = 0, total = 0;
    cudaCheck(cudaMemGetInfo(&free0, &total), "memGetInfo");
    Impl& d = *d_;
    d.dm = buildDeviceMesh(cs.m, cs.g, cs.patches);
    if (d.dm.nBndFaces != cs.nB || d.dm.nInternalFaces != cs.nIF)
        throw std::runtime_error("brae rhoCentralFoam: the device mesh's face layout is not the case's.");
    const int nC = cs.nC, nB = cs.nB, nIF = cs.nIF;
    up(d.rho, s0.rho);
    upCmpt(d.rhoUx, s0.rhoU, 0);
    upCmpt(d.rhoUy, s0.rhoU, 1);
    upCmpt(d.rhoUz, s0.rhoU, 2);
    up(d.rhoE, s0.rhoE);
    up(d.e, s0.e);
    up(d.T, s0.T);
    up(d.p, s0.p);
    up(d.psi, s0.psi);
    upCmpt(d.Ux, s0.U, 0);
    upCmpt(d.Uy, s0.U, 1);
    upCmpt(d.Uz, s0.U, 2);
    d.Told.resize(nC);
    d.rPsi.resize(nC);
    d.c.resize(nC);
    d.coCell.resize(nC);
    up(d.rhoB, s0.rhoB);
    upCmpt(d.rhoUxB, s0.rhoUB, 0);
    upCmpt(d.rhoUyB, s0.rhoUB, 1);
    upCmpt(d.rhoUzB, s0.rhoUB, 2);
    up(d.rhoEB, s0.rhoEB);
    up(d.eB, s0.eB);
    up(d.TB, s0.TB);
    up(d.pB, s0.pB);
    up(d.psiB, s0.psiB);
    upCmpt(d.UxB, s0.UB, 0);
    upCmpt(d.UyB, s0.UB, 1);
    upCmpt(d.UzB, s0.UB, 2);
    d.rPsiB.resize(nB);
    d.cB.resize(nB);
    d.bCell.copyFrom(cs.bCell);
    {
        std::vector<int> kU(nB), kP(nB), kT(nB);
        for (label k = 0; k < nB; ++k)
        {
            kU[k] = cs.bcU[cs.bPatch[k]];
            kP[k] = cs.bcP[cs.bPatch[k]];
            kT[k] = cs.bcT[cs.bPatch[k]];
        }
        d.kU.copyFrom(kU);
        d.kP.copyFrom(kP);
        d.kT.copyFrom(kT);
    }
    for (int i = 0; i < 7; ++i)
    {
        d.gx[i].resize(nC);
        d.gy[i].resize(nC);
        d.gz[i].resize(nC);
    }
    for (DeviceBuffer<scalar>* b : {&d.phi, &d.phiUx, &d.phiUy, &d.phiUz, &d.phiEp, &d.amaxSf}) b->resize(nIF);
    for (DeviceBuffer<scalar>* b : {&d.phiB, &d.phiUxB, &d.phiUyB, &d.phiUzB, &d.phiEpB, &d.amaxSfB}) b->resize(nB);
    d.status.copyFrom(std::vector<int>(kReportWindow, 0));
    d.running.copyFrom(std::vector<double>(5, 0.0));
    d.courant.copyFrom(std::vector<double>(3*kReportWindow, 0.0));
    cudaCheck(cudaDeviceSynchronize(), "rhoCentralFoam upload");
    size_t free1 = 0;
    cudaCheck(cudaMemGetInfo(&free1, &total), "memGetInfo");
    bytes_ = (free0 > free1) ? (free0 - free1) : 0;
}

DeviceSolver::~DeviceSolver() = default;


void DeviceSolver::step()
{
    if (pending_ >= kReportWindow)
        throw std::logic_error("brae rhoCentralFoam: DeviceSolver::step with a full report window; flush() first.");
    Impl& d = *d_;
    const Case& cs = cs_;
    const int nC = cs.nC, nB = cs.nB, nIF = cs.nIF;
    const scalar CpByCv = gasCp(cs.gas)/gasCv(cs.gas);

    auxKernel<<<blocksFor(nC), kTPB>>>(nC, CpByCv, d.psi.data(), d.rPsi.data(), d.c.data());
    if (nB) auxKernel<<<blocksFor(nB), kTPB>>>(nB, CpByCv, d.psiB.data(), d.rPsiB.data(), d.cB.data());
    cudaCheck(cudaGetLastError(), "rhoCentral aux");

    // the limiters' gradients, three fields a launch (deviceGaussGradFused, bit-identical to one at a time)
    {
        const DeviceBuffer<scalar>* vol[7] = {&d.rho, &d.rhoUx, &d.rhoUy, &d.rhoUz, &d.rPsi, &d.e, &d.c};
        const DeviceBuffer<scalar>* bv[7] = {&d.rhoB, &d.rhoUxB, &d.rhoUyB, &d.rhoUzB, &d.rPsiB, &d.eB, &d.cB};
        deviceGaussGradFused(d.dm, 3, vol, bv, d.gx, d.gy, d.gz, nullptr);
        deviceGaussGradFused(d.dm, 3, vol + 3, bv + 3, d.gx + 3, d.gy + 3, d.gz + 3, nullptr);
        deviceGaussGradFused(d.dm, 1, vol + 6, bv + 6, d.gx + 6, d.gy + 6, d.gz + 6, nullptr);
    }
    Grads G;
    for (int i = 0; i < 7; ++i)
    {
        G.x[i] = d.gx[i].data();
        G.y[i] = d.gy[i].data();
        G.z[i] = d.gz[i].data();
    }
    if (nIF)
    {
        CellFields F{d.rho.data(), d.rhoUx.data(), d.rhoUy.data(), d.rhoUz.data(), d.rPsi.data(), d.e.data(),
                     d.c.data()};
        faceKernel<<<blocksFor(nIF), kTPB>>>(nIF, cs.tadmor, control_, d.dm.owner.data(), d.dm.nei.data(), d.dm.w.data(),
                                             d.dm.Sfx.data(), d.dm.Sfy.data(), d.dm.Sfz.data(), d.dm.magSf.data(),
                                             d.dm.Cx.data(), d.dm.Cy.data(), d.dm.Cz.data(), F, G,
                                             d.phi.data(), d.phiUx.data(), d.phiUy.data(), d.phiUz.data(),
                                             d.phiEp.data(), d.amaxSf.data());
        cudaCheck(cudaGetLastError(), "rhoCentral faces");
    }
    if (nB)
    {
        BndFields B{d.rhoB.data(), d.rhoUxB.data(), d.rhoUyB.data(), d.rhoUzB.data(), d.rPsiB.data(), d.eB.data(),
                    d.cB.data()};
        bndFaceKernel<<<blocksFor(nB), kTPB>>>(nB, nIF, cs.tadmor, control_, cs.deltaT, d.dm.bndIsEmpty.data(),
                                               d.dm.Sfx.data(), d.dm.Sfy.data(), d.dm.Sfz.data(), d.dm.magSf.data(),
                                               B, d.phiB.data(), d.phiUxB.data(), d.phiUyB.data(), d.phiUzB.data(),
                                               d.phiEpB.data(), d.amaxSfB.data(), d.running.data());
        cudaCheck(cudaGetLastError(), "rhoCentral patch faces");
    }
    {
        Fluxes Fx{d.phi.data(), d.phiUx.data(), d.phiUy.data(), d.phiUz.data(), d.phiEp.data(), d.amaxSf.data(),
                  d.phiB.data(), d.phiUxB.data(), d.phiUyB.data(), d.phiUzB.data(), d.phiEpB.data(),
                  d.amaxSfB.data()};
        CellState S{d.rho.data(), d.rhoUx.data(), d.rhoUy.data(), d.rhoUz.data(), d.rhoE.data(), d.e.data(),
                    d.T.data(), d.Told.data(), d.p.data(), d.psi.data(), d.Ux.data(), d.Uy.data(), d.Uz.data()};
        cellKernel<<<blocksFor(nC), kTPB>>>(nC, pending_, 1.0/cs.deltaT, cs.sd.valid(0), cs.sd.valid(1), cs.sd.valid(2),
                                            cs.gas, d.dm.ownerStart.data(), d.dm.losort.data(),
                                            d.dm.losortStart.data(), d.dm.bndCellStart.data(), d.dm.bndPerm.data(),
                                            d.dm.bndIsEmpty.data(), d.dm.V.data(), Fx, S, d.coCell.data(),
                                            d.status.data());
        cudaCheck(cudaGetLastError(), "rhoCentral cells");
    }
    if (nB)
    {
        CellRead C{d.Ux.data(), d.Uy.data(), d.Uz.data(), d.e.data(), d.Told.data(), d.p.data()};
        BndState B{d.rhoB.data(), d.rhoUxB.data(), d.rhoUyB.data(), d.rhoUzB.data(), d.rhoEB.data(), d.eB.data(),
                   d.TB.data(), d.pB.data(), d.psiB.data(), d.UxB.data(), d.UyB.data(), d.UzB.data()};
        bndKernel<<<blocksFor(nB), kTPB>>>(nB, pending_, cs.gas, d.bCell.data(), d.kU.data(), d.kP.data(), d.kT.data(), C, B,
                                           d.status.data());
        cudaCheck(cudaGetLastError(), "rhoCentral patches");
    }
    courantKernel<<<1, 1024>>>(nC, d.coCell.data(), d.dm.V.data(), d.courant.data() + 3*pending_);
    cudaCheck(cudaGetLastError(), "rhoCentral courant");
    ++pending_;
}


std::vector<StepReport> DeviceSolver::flush()
{
    Impl& d = *d_;
    std::vector<StepReport> out(pending_);
    if (!pending_) return out;
    std::vector<double> co(3*pending_);
    std::vector<int> st(pending_);
    cudaCheck(cudaMemcpy(co.data(), d.courant.data(), co.size()*sizeof(double), cudaMemcpyDeviceToHost),
              "courant D2H");
    cudaCheck(cudaMemcpy(st.data(), d.status.data(), st.size()*sizeof(int), cudaMemcpyDeviceToHost), "status D2H");
    for (int i = 0; i < pending_; ++i)
    {
        if (cs_.nIF)
        {
            out[i].CoNum = 0.5*co[3*i]*cs_.deltaT;
            out[i].meanCoNum = 0.5*(co[3*i + 1]/co[3*i + 2])*cs_.deltaT;
        }
        out[i].status = st[i];
    }
    cudaCheck(cudaMemset(d.status.data(), 0, st.size()*sizeof(int)), "status reset");
    pending_ = 0;
    return out;
}


void DeviceSolver::download(State& s) const
{
    const Impl& d = *d_;
    s.rho = d.rho.host();
    downCmpt(d.rhoUx, s.rhoU, 0);
    downCmpt(d.rhoUy, s.rhoU, 1);
    downCmpt(d.rhoUz, s.rhoU, 2);
    s.rhoE = d.rhoE.host();
    s.e = d.e.host();
    s.T = d.T.host();
    s.p = d.p.host();
    s.psi = d.psi.host();
    downCmpt(d.Ux, s.U, 0);
    downCmpt(d.Uy, s.U, 1);
    downCmpt(d.Uz, s.U, 2);
    s.rhoB = d.rhoB.host();
    downCmpt(d.rhoUxB, s.rhoUB, 0);
    downCmpt(d.rhoUyB, s.rhoUB, 1);
    downCmpt(d.rhoUzB, s.rhoUB, 2);
    s.rhoEB = d.rhoEB.host();
    s.eB = d.eB.host();
    s.TB = d.TB.host();
    s.pB = d.pB.host();
    s.psiB = d.psiB.host();
    downCmpt(d.UxB, s.UB, 0);
    downCmpt(d.UyB, s.UB, 1);
    downCmpt(d.UzB, s.UB, 2);
}


void DeviceSolver::boundaryIntegrals(scalar& mass, vector& mom, scalar& energy) const
{
    const std::vector<double> r = d_->running.host();
    mass = r[0];
    mom = vector{r[1], r[2], r[3]};
    energy = r[4];
}

} // namespace rhoCentral
} // namespace brae
