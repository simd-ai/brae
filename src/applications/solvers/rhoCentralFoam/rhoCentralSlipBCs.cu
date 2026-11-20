// rhoCentralSlipBCs.cu -- maxwellSlipU and smoluchowskiJumpT on the CUDA arm (milestone I). See the header.
#include "rhoCentralSlipBCs.cuh"
#include "rhoCentralThermo.cuh"

#include "device_buffer.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

namespace {

constexpr int kTPB = 256;
inline int blocksFor(int n) { return (n + kTPB - 1)/kTPB; }

// constant::mathematical::piByTwo = 0.5*pi
constexpr scalar kPiByTwo = 1.5707963267948966;

// maxwellSlipUFvPatchVectorField::updateCoeffs (:120-170), one face. Each line is one of OpenFOAM's field
// operations, so each is rounded where OpenFOAM stores a field:
//   C1 = sqrt(psi*piByTwo)*(2 - a)/a;  nu = mu/rho;  vf = 1/(1 + deltaCoeffs*C1*nu);  R = Uwall
//   thermalCreep: R -= 3*nu/(4*T_b)*transform(I - n n, grad(T)_b)
//   curvature:    R -= C1/rho*transform(I - n n, n & tauMC_b)
// grad(T)_b is gaussGrad's patch value: the face cell's gradient plus n*(snGrad(T) - n & it) (gaussGrad.C:135-160);
// T's snGrad is taken as (T_b - T_c)*deltaCoeffs, whose normal part the (I - n n) removes.
__global__ void maxwellKernel(int n, const label* __restrict__ face, const label* __restrict__ cell,
                              const scalar* __restrict__ nx, const scalar* __restrict__ ny,
                              const scalar* __restrict__ nz, const scalar* __restrict__ dc,
                              const scalar* __restrict__ acc, const scalar* __restrict__ uwx,
                              const scalar* __restrict__ uwy, const scalar* __restrict__ uwz,
                              const int* __restrict__ creep, const int* __restrict__ curv, GasCoeffs gas,
                              const scalar* __restrict__ T, const scalar* __restrict__ TB,
                              const scalar* __restrict__ rhoB, const scalar* __restrict__ psiB,
                              const scalar* __restrict__ gx, const scalar* __restrict__ gy,
                              const scalar* __restrict__ gz, const scalar* __restrict__ tauB, int nB,
                              scalar* __restrict__ vf, scalar* __restrict__ rx, scalar* __restrict__ ry,
                              scalar* __restrict__ rz)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label k = face[i];
    const label c = cell[i];
    const scalar a = acc[i];
    const scalar mu = gasMu(TB[k], gas);                 // thermo:mu's patch value: mu(T_b) (hePsiThermo::calculate)
    const scalar sq = sqrt(__dmul_rn(psiB[k], kPiByTwo));
    const scalar C1 = __dmul_rn(sq, 2.0 - a)/a;
    const scalar nu = mu/rhoB[k];
    vf[i] = 1.0/(1.0 + __dmul_rn(__dmul_rn(dc[i], C1), nu));
    const scalar n0 = nx[i], n1 = ny[i], n2 = nz[i];
    // I - n*n, each product rounded (a tensor field), then the transform t & v
    const scalar txx = 1 - __dmul_rn(n0, n0), txy = 0 - __dmul_rn(n0, n1), txz = 0 - __dmul_rn(n0, n2);
    const scalar tyx = 0 - __dmul_rn(n1, n0), tyy = 1 - __dmul_rn(n1, n1), tyz = 0 - __dmul_rn(n1, n2);
    const scalar tzx = 0 - __dmul_rn(n2, n0), tzy = 0 - __dmul_rn(n2, n1), tzz = 1 - __dmul_rn(n2, n2);
    scalar R0 = uwx[i], R1 = uwy[i], R2 = uwz[i];
    if (creep[i])
    {
        const scalar g0 = gx[c], g1 = gy[c], g2 = gz[c];
        const scalar sn = __dmul_rn(TB[k] - T[c], dc[i]);
        const scalar d = sn - (n0*g0 + n1*g1 + n2*g2);
        const scalar p0 = g0 + __dmul_rn(n0, d), p1 = g1 + __dmul_rn(n1, d), p2 = g2 + __dmul_rn(n2, d);
        const scalar t0 = txx*p0 + txy*p1 + txz*p2;
        const scalar t1 = tyx*p0 + tyy*p1 + tyz*p2;
        const scalar t2 = tzx*p0 + tzy*p1 + tzz*p2;
        const scalar f = __dmul_rn(3.0, nu)/__dmul_rn(4.0, TB[k]);
        R0 = R0 - __dmul_rn(f, t0);
        R1 = R1 - __dmul_rn(f, t1);
        R2 = R2 - __dmul_rn(f, t2);
    }
    if (curv[i])
    {
        auto tau = [&](int r, int s) { return tauB[(r*3 + s)*nB + k]; };
        const scalar v0 = n0*tau(0, 0) + n1*tau(1, 0) + n2*tau(2, 0);   // n & tauMC_b
        const scalar v1 = n0*tau(0, 1) + n1*tau(1, 1) + n2*tau(2, 1);
        const scalar v2 = n0*tau(0, 2) + n1*tau(1, 2) + n2*tau(2, 2);
        const scalar w0 = txx*v0 + txy*v1 + txz*v2;
        const scalar w1 = tyx*v0 + tyy*v1 + tyz*v2;
        const scalar w2 = tzx*v0 + tzy*v1 + tzz*v2;
        const scalar f = C1/rhoB[k];
        R0 = R0 - __dmul_rn(f, w0);
        R1 = R1 - __dmul_rn(f, w1);
        R2 = R2 - __dmul_rn(f, w2);
    }
    rx[i] = R0;
    ry[i] = R1;
    rz[i] = R2;
}

// partialSlip's value from the cells' U: lerp(transform(I - n n, U_c), R, vf), (1 - vf)*a + t*b per component
__device__ __forceinline__ void partialSlipValue(scalar n0, scalar n1, scalar n2, scalar u0, scalar u1, scalar u2,
                                                 scalar v, scalar R0, scalar R1, scalar R2, scalar& b0, scalar& b1,
                                                 scalar& b2)
{
    const scalar txx = 1 - __dmul_rn(n0, n0), txy = 0 - __dmul_rn(n0, n1), txz = 0 - __dmul_rn(n0, n2);
    const scalar tyx = 0 - __dmul_rn(n1, n0), tyy = 1 - __dmul_rn(n1, n1), tyz = 0 - __dmul_rn(n1, n2);
    const scalar tzx = 0 - __dmul_rn(n2, n0), tzy = 0 - __dmul_rn(n2, n1), tzz = 1 - __dmul_rn(n2, n2);
    const scalar a0 = txx*u0 + txy*u1 + txz*u2;
    const scalar a1 = tyx*u0 + tyy*u1 + tyz*u2;
    const scalar a2 = tzx*u0 + tzy*u1 + tzz*u2;
    const scalar w = 1 - v;
    b0 = __dmul_rn(w, a0) + __dmul_rn(v, R0);
    b1 = __dmul_rn(w, a1) + __dmul_rn(v, R1);
    b2 = __dmul_rn(w, a2) + __dmul_rn(v, R2);
}

// brae's per-component mixed form of a partialSlip face: valueFraction_k = vf + (1 - vf)*|n_k|
// (snGradTransformDiag), refValue_k = U_c,k + (U_b,k - U_c,k)/diag_k, so the mixed coefficients -deltaCoeffs*diag_k
// and deltaCoeffs*diag_k*refValue_k are partialSlip's -deltaCoeffs*diag_k and snGrad_k + deltaCoeffs*diag_k*U_c,k
__global__ void emulateKernel(int n, const label* __restrict__ face, const label* __restrict__ cell,
                              const scalar* __restrict__ nx, const scalar* __restrict__ ny,
                              const scalar* __restrict__ nz, const scalar* __restrict__ vf,
                              const scalar* __restrict__ rx, const scalar* __restrict__ ry,
                              const scalar* __restrict__ rz, const scalar* __restrict__ Ux,
                              const scalar* __restrict__ Uy, const scalar* __restrict__ Uz,
                              scalar* __restrict__ vf0, scalar* __restrict__ vf1, scalar* __restrict__ vf2,
                              scalar* __restrict__ r0, scalar* __restrict__ r1, scalar* __restrict__ r2)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label k = face[i];
    const label c = cell[i];
    const scalar v = vf[i];
    const scalar u0 = Ux[c], u1 = Uy[c], u2 = Uz[c];
    scalar b0, b1, b2;
    partialSlipValue(nx[i], ny[i], nz[i], u0, u1, u2, v, rx[i], ry[i], rz[i], b0, b1, b2);
    const scalar d0 = v + __dmul_rn(1 - v, fabs(nx[i]));
    const scalar d1 = v + __dmul_rn(1 - v, fabs(ny[i]));
    const scalar d2 = v + __dmul_rn(1 - v, fabs(nz[i]));
    vf0[k] = d0;
    vf1[k] = d1;
    vf2[k] = d2;
    r0[k] = u0 + (b0 - u0)/d0;
    r1[k] = u1 + (b1 - u1)/d1;
    r2[k] = u2 + (b2 - u2)/d2;
}

__global__ void evaluateKernel(int n, const label* __restrict__ face, const label* __restrict__ cell,
                               const scalar* __restrict__ nx, const scalar* __restrict__ ny,
                               const scalar* __restrict__ nz, const scalar* __restrict__ vf,
                               const scalar* __restrict__ rx, const scalar* __restrict__ ry,
                               const scalar* __restrict__ rz, const scalar* __restrict__ Ux,
                               const scalar* __restrict__ Uy, const scalar* __restrict__ Uz,
                               scalar* __restrict__ UxB, scalar* __restrict__ UyB, scalar* __restrict__ UzB)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label k = face[i];
    const label c = cell[i];
    partialSlipValue(nx[i], ny[i], nz[i], Ux[c], Uy[c], Uz[c], vf[i], rx[i], ry[i], rz[i], UxB[k], UyB[k], UzB[k]);
}

// smoluchowskiJumpTFvPatchScalarField::updateCoeffs (:122-162): C2 = mu/rho*sqrt(psi*piByTwo)*2*gamma/Pr/(gamma + 1)
// *(2 - a)/a, left to right; valueFraction = 1/(1 + deltaCoeffs*C2). mixedEnergy (:96-125): e's refValue
// he(p_b, Twall) (its refGrad is Cpv*0 + deltaCoeffs*(he - he) = 0 on a pureMixture)
__global__ void smoluchowskiKernel(int n, const label* __restrict__ face, const scalar* __restrict__ dc,
                                   const scalar* __restrict__ acc, const scalar* __restrict__ Tw,
                                   const scalar* __restrict__ gamma, scalar Pr, GasCoeffs gas,
                                   const scalar* __restrict__ TB, const scalar* __restrict__ rhoB,
                                   const scalar* __restrict__ psiB, const scalar* __restrict__ pB,
                                   scalar* __restrict__ vfT, scalar* __restrict__ eRef, int noJump)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label k = face[i];
    const scalar a = acc[i];
    const scalar g = gamma[i];
    const scalar mu = gasMu(TB[k], gas);
    scalar C2 = mu/rhoB[k];
    C2 = __dmul_rn(C2, sqrt(__dmul_rn(psiB[k], kPiByTwo)));
    C2 = __dmul_rn(C2, 2.0);
    C2 = __dmul_rn(C2, g);
    C2 = C2/Pr;
    C2 = C2/(g + 1.0);
    C2 = __dmul_rn(C2, 2.0 - a);
    C2 = C2/a;
    vfT[k] = noJump ? scalar(1) : 1.0/(1.0 + __dmul_rn(dc[i], C2));
    eRef[k] = gasEs(pB[k], Tw[i], gas);
}

__global__ void gatherKernel(int n, const label* __restrict__ face, const scalar* __restrict__ src,
                             scalar* __restrict__ dst)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[face[i]];
}
__global__ void scatterKernel(int n, const label* __restrict__ face, const scalar* __restrict__ src,
                              scalar* __restrict__ dst)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[face[i]] = src[i];
}

} // namespace

SlipWalls::SlipWalls(const Case& cs, const DeviceMesh& dm, DeviceVectorBoundary& dbU, DeviceBoundary& dbT, int control)
    : cs_(cs), dm_(dm), nMs_(static_cast<int>(cs.msFaces.size())), nSj_(static_cast<int>(cs.sjFaces.size())),
      noJump_(control == controlSmoluchowskiNoJump ? 1 : 0)
{
    const int nB = cs.nB;
    const std::vector<vector>& Sf = cs.g.Sf();
    const std::vector<scalar>& magSf = cs.g.magSf();
    std::vector<scalar> dcAll = dbU.comp[0].deltaCoeffs.host();   // patch().deltaCoeffs(), per boundary face
    if (nMs_)
    {
        std::vector<label> cell(nMs_);
        std::vector<scalar> n[3], dc(nMs_), uw[3], r[3];
        for (int j = 0; j < 3; ++j) { n[j].resize(nMs_); uw[j].resize(nMs_); r[j].resize(nMs_); }
        for (int i = 0; i < nMs_; ++i)
        {
            const label k = cs.msFaces[i];
            const label f = cs.nIF + k;   // no coupled patch is accepted: flat face k is mesh face nInternalFaces + k
            cell[i] = cs.bCell[k];
            // patch().nf() = Sf/magSf
            n[0][i] = Sf[f].x/magSf[f];
            n[1][i] = Sf[f].y/magSf[f];
            n[2][i] = Sf[f].z/magSf[f];
            dc[i] = dcAll[static_cast<std::size_t>(k)];
            uw[0][i] = cs.msUwall[i].x; uw[1][i] = cs.msUwall[i].y; uw[2][i] = cs.msUwall[i].z;
            r[0][i] = cs.msRef0[i].x; r[1][i] = cs.msRef0[i].y; r[2][i] = cs.msRef0[i].z;
        }
        msFaces_.copyFrom(cs.msFaces);
        msCell_.copyFrom(cell);
        for (int j = 0; j < 3; ++j) { msN_[j].copyFrom(n[j]); msUw_[j].copyFrom(uw[j]); msR_[j].copyFrom(r[j]); }
        msDc_.copyFrom(dc);
        msAcc_.copyFrom(cs.msAcc);
        msCreep_.copyFrom(control == controlMaxwellNoCreep ? std::vector<int>(nMs_, 0) : cs.msCreep);
        msCurv_.copyFrom(control == controlMaxwellNoCurvature ? std::vector<int>(nMs_, 0) : cs.msCurv);
        // the dictionary constructor's refValue and valueFraction, until the first updateCoeffs
        msVf_.copyFrom(cs.msVf0);
        // the U descriptor: each component a mixed face (its coefficients from emulate())
        for (int j = 0; j < 3; ++j)
        {
            std::vector<label> ty = dbU.comp[j].bcType.host();
            for (label k : cs.msFaces) ty[static_cast<std::size_t>(k)] = 5;
            dbU.comp[j].bcType.copyFrom(ty);
            if (dbU.comp[j].valueFraction.size() != static_cast<std::size_t>(nB))
                dbU.comp[j].valueFraction.copyFrom(std::vector<scalar>(nB, 0.0));
            if (dbU.comp[j].refValue.size() != static_cast<std::size_t>(nB))
                dbU.comp[j].refValue.copyFrom(std::vector<scalar>(nB, 0.0));
        }
        gx_.resize(cs.nC);
        gy_.resize(cs.nC);
        gz_.resize(cs.nC);
    }
    if (nSj_)
    {
        std::vector<scalar> dc(nSj_);
        for (int i = 0; i < nSj_; ++i) dc[i] = dcAll[static_cast<std::size_t>(cs.sjFaces[i])];
        sjFaces_.copyFrom(cs.sjFaces);
        sjDc_.copyFrom(dc);
        sjAcc_.copyFrom(cs.sjAcc);
        sjTw_.copyFrom(cs.sjTwall);
        sjGamma_.copyFrom(cs.sjGamma);
        sjTb_.resize(nSj_);
        // T's descriptor: mixed, refValue Twall, refGrad 0, valueFraction 0 (the dictionary constructor's)
        std::vector<label> ty = dbT.bcType.host();
        std::vector<scalar> rv = dbT.refValue.size() == static_cast<std::size_t>(nB) ? dbT.refValue.host()
                                                                                       : std::vector<scalar>(nB, 0.0);
        std::vector<scalar> vf = dbT.valueFraction.size() == static_cast<std::size_t>(nB)
                                     ? dbT.valueFraction.host() : std::vector<scalar>(nB, 0.0);
        std::vector<scalar> rg = dbT.refGrad.size() == static_cast<std::size_t>(nB) ? dbT.refGrad.host()
                                                                                      : std::vector<scalar>(nB, 0.0);
        for (int i = 0; i < nSj_; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(cs.sjFaces[i]);
            ty[k] = 5;
            rv[k] = cs.sjTwall[i];
            vf[k] = 0;
            rg[k] = 0;
        }
        dbT.bcType.copyFrom(ty);
        dbT.refValue.copyFrom(rv);
        dbT.valueFraction.copyFrom(vf);
        dbT.refGrad.copyFrom(rg);
        eRef_.copyFrom(std::vector<scalar>(nB, 0.0));
    }
}

void SlipWalls::updateMaxwell(const DeviceBuffer<scalar>& T, const DeviceBuffer<scalar>& TB,
                              const DeviceBuffer<scalar>& rhoB, const DeviceBuffer<scalar>& psiB,
                              const DeviceBuffer<scalar>& tauB)
{
    if (!nMs_) return;
    // fvc::grad(T): Gauss linear on the cells and T's patch values as they stand
    const DeviceBuffer<scalar>* vol[1] = {&T};
    const DeviceBuffer<scalar>* bv[1] = {&TB};
    deviceGaussGradFused(dm_, 1, vol, bv, &gx_, &gy_, &gz_);
    maxwellKernel<<<blocksFor(nMs_), kTPB>>>(nMs_, msFaces_.data(), msCell_.data(), msN_[0].data(), msN_[1].data(),
                                             msN_[2].data(), msDc_.data(), msAcc_.data(), msUw_[0].data(),
                                             msUw_[1].data(), msUw_[2].data(), msCreep_.data(), msCurv_.data(),
                                             cs_.gas, T.data(), TB.data(), rhoB.data(), psiB.data(), gx_.data(),
                                             gy_.data(), gz_.data(), tauB.data(), cs_.nB, msVf_.data(),
                                             msR_[0].data(), msR_[1].data(), msR_[2].data());
    cudaCheck(cudaGetLastError(), "maxwellSlipU updateCoeffs");
}

void SlipWalls::emulate(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                        const DeviceBuffer<scalar>& Uz)
{
    if (!nMs_) return;
    emulateKernel<<<blocksFor(nMs_), kTPB>>>(nMs_, msFaces_.data(), msCell_.data(), msN_[0].data(), msN_[1].data(),
                                             msN_[2].data(), msVf_.data(), msR_[0].data(), msR_[1].data(),
                                             msR_[2].data(), Ux.data(), Uy.data(), Uz.data(),
                                             dbU.comp[0].valueFraction.data(), dbU.comp[1].valueFraction.data(),
                                             dbU.comp[2].valueFraction.data(), dbU.comp[0].refValue.data(),
                                             dbU.comp[1].refValue.data(), dbU.comp[2].refValue.data());
    cudaCheck(cudaGetLastError(), "partialSlip coefficients");
}

void SlipWalls::evaluate(const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                         DeviceBuffer<scalar>& UxB, DeviceBuffer<scalar>& UyB, DeviceBuffer<scalar>& UzB)
{
    if (!nMs_) return;
    evaluateKernel<<<blocksFor(nMs_), kTPB>>>(nMs_, msFaces_.data(), msCell_.data(), msN_[0].data(), msN_[1].data(),
                                              msN_[2].data(), msVf_.data(), msR_[0].data(), msR_[1].data(),
                                              msR_[2].data(), Ux.data(), Uy.data(), Uz.data(), UxB.data(),
                                              UyB.data(), UzB.data());
    cudaCheck(cudaGetLastError(), "partialSlip evaluate");
}

void SlipWalls::updateSmoluchowski(const DeviceBuffer<scalar>& TB, const DeviceBuffer<scalar>& rhoB,
                                   const DeviceBuffer<scalar>& psiB, const DeviceBuffer<scalar>& pB,
                                   DeviceBoundary& dbT)
{
    if (!nSj_) return;
    smoluchowskiKernel<<<blocksFor(nSj_), kTPB>>>(nSj_, sjFaces_.data(), sjDc_.data(), sjAcc_.data(), sjTw_.data(),
                                                  sjGamma_.data(), cs_.sjPr, cs_.gas, TB.data(), rhoB.data(),
                                                  psiB.data(), pB.data(), dbT.valueFraction.data(), eRef_.data(),
                                                  noJump_);
    cudaCheck(cudaGetLastError(), "smoluchowskiJumpT updateCoeffs");
}

void SlipWalls::keepTb(const DeviceBuffer<scalar>& TbEval)
{
    if (!nSj_) return;
    gatherKernel<<<blocksFor(nSj_), kTPB>>>(nSj_, sjFaces_.data(), TbEval.data(), sjTb_.data());
    cudaCheck(cudaGetLastError(), "smoluchowskiJumpT keep T_b");
}

void SlipWalls::restoreTb(DeviceBuffer<scalar>& TbEval) const
{
    if (!nSj_) return;
    scatterKernel<<<blocksFor(nSj_), kTPB>>>(nSj_, sjFaces_.data(), sjTb_.data(), TbEval.data());
    cudaCheck(cudaGetLastError(), "smoluchowskiJumpT restore T_b");
}

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
