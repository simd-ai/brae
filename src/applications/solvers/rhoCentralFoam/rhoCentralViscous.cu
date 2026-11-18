// rhoCentralViscous.cu -- see the header. Every expression is rhoCentralFoam.C's; a product OpenFOAM stores in a field
// before summing it is rounded on its own (__dmul_rn), never fused into the sum.
#include "rhoCentralViscous.cuh"
#include "rhoCentralPatchUpdate.cuh"

#include "device_amg.cuh"
#include "device_blas.cuh"
#include "device_ddt.cuh"
#include "device_divdevreff.cuh"
#include "device_pcg.cuh"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

namespace {

constexpr int kTPB = 256;
inline int blocksFor(int n) { return (n + kTPB - 1)/kTPB; }

void copyD2D(DeviceBuffer<scalar>& dst, const DeviceBuffer<scalar>& src)
{
    dst.resize(src.size());
    if (src.size())
        cudaCheck(cudaMemcpyAsync(dst.data(), src.data(), src.size()*sizeof(scalar), cudaMemcpyDeviceToDevice,
                                  cudaStreamPerThread), "viscous copy");
}
void copySlice(DeviceBuffer<scalar>& dst, int slot, int n, const DeviceBuffer<scalar>& src)
{
    if (n)
        cudaCheck(cudaMemcpyAsync(dst.data() + static_cast<std::size_t>(slot)*n, src.data(), n*sizeof(scalar),
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread), "grad(U) slice");
}

__global__ void negateKernel(int n, const scalar* __restrict__ a, scalar* __restrict__ b)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) b[i] = -a[i];
}
__global__ void scaleKernel(int n, scalar s, const scalar* __restrict__ a, scalar* __restrict__ b)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) b[i] = s*a[i];   // gammaMagSf = gamma_f*magSf, gamma_f = linear interpolate(const) = const
}

// fvm::ddt - fvc::ddt (already in diag and source) minus fvm::laplacian: diag -= lap.diag (lduMatrix::operator-=),
// source -= lap.source (the corrected scheme's explicit part, when the mesh has one), then - fvc::div(tauMC):
// source += V*div (fvMatrix's operator- of a field)
__global__ void finishMatrixKernel(int n, const scalar* __restrict__ diagD, const scalar* __restrict__ lapDiag,
                                   const scalar* __restrict__ lapCorr, const scalar* __restrict__ V,
                                   const scalar* __restrict__ divT, scalar* __restrict__ diagM,
                                   scalar* __restrict__ src)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    if (diagM) diagM[c] = diagD[c] - lapDiag[c];
    scalar s = src[c];
    if (lapCorr) s = s - lapCorr[c];
    if (divT) s = s + __dmul_rn(V[c], divT[c]);
    src[c] = s;
}

__global__ void rhoTimesKernel(int n, const scalar* __restrict__ rho, const scalar* __restrict__ u,
                               scalar* __restrict__ out)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n) out[c] = rho[c]*u[c];   // rhoU = rho*U
}

// fvc::snGrad(U) on the patch faces, the patch fields' snGrad(): fixedValue (and inletOutlet on inflow, mixed with
// valueFraction 1) deltaCoeffs*(U_b - U_c); zeroGradient (inletOutlet on outflow) 0; symmetryPlane, symmetry and slip
// (basicSymmetryFvPatchField.C snGrad) (transform(I - 2 n n, U_c) - U_c)*(deltaCoeffs/2). Empty faces: 0.
__global__ void snGradPatchKernel(int nB, const label* __restrict__ isEmpty, const int* __restrict__ kind,
                                  const tensor* __restrict__ R, const label* __restrict__ bCell,
                                  const scalar* __restrict__ dc, const label* __restrict__ t0,
                                  const label* __restrict__ t1, const label* __restrict__ t2,
                                  const scalar* __restrict__ Ux, const scalar* __restrict__ Uy,
                                  const scalar* __restrict__ Uz, const scalar* __restrict__ UxB,
                                  const scalar* __restrict__ UyB, const scalar* __restrict__ UzB,
                                  scalar* __restrict__ sx, scalar* __restrict__ sy, scalar* __restrict__ sz)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    if (isEmpty[k])
    {
        sx[k] = sy[k] = sz[k] = 0;
        return;
    }
    const label c = bCell[k];
    const vector u{Ux[c], Uy[c], Uz[c]};
    if (kind[k] == transformMirror)
    {
        const tensor& r = R[k];
        const vector m{r.xx*u.x + r.xy*u.y + r.xz*u.z, r.yx*u.x + r.yy*u.y + r.yz*u.z,
                       r.zx*u.x + r.zy*u.y + r.zz*u.z};
        const scalar h = dc[k]/2.0;
        sx[k] = (m.x - u.x)*h;
        sy[k] = (m.y - u.y)*h;
        sz[k] = (m.z - u.z)*h;
        return;
    }
    sx[k] = (t0[k] == 1) ? dc[k]*(UxB[k] - u.x) : scalar(0);
    sy[k] = (t1[k] == 1) ? dc[k]*(UyB[k] - u.y) : scalar(0);
    sz[k] = (t2[k] == 1) ? dc[k]*(UzB[k] - u.z) : scalar(0);
}

// sigmaDotU on one face: (muf*magSf*snGrad(U) + Sf & tauMC_f) & aU. muf*magSf and each product are fields of
// their own in OpenFOAM, rounded alone; the inner product is Vector's operator& (one expression).
__device__ __forceinline__ scalar sigmaFace(scalar muMagSf, scalar s0, scalar s1, scalar s2, const scalar* d,
                                           scalar a0, scalar a1, scalar a2)
{
    scalar v0 = __dmul_rn(muMagSf, s0), v1 = __dmul_rn(muMagSf, s1), v2 = __dmul_rn(muMagSf, s2);
    if (d)
    {
        v0 = v0 + d[0];
        v1 = v1 + d[1];
        v2 = v2 + d[2];
    }
    return v0*a0 + v1*a1 + v2*a2;
}
__global__ void sigmaInternalKernel(int nIf, scalar mu, const scalar* __restrict__ magSf,
                                    const scalar* __restrict__ s0, const scalar* __restrict__ s1,
                                    const scalar* __restrict__ s2, const scalar* __restrict__ d0,
                                    const scalar* __restrict__ d1, const scalar* __restrict__ d2,
                                    const scalar* __restrict__ a0, const scalar* __restrict__ a1,
                                    const scalar* __restrict__ a2, scalar* __restrict__ out)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    scalar d[3];
    if (d0) { d[0] = d0[f]; d[1] = d1[f]; d[2] = d2[f]; }
    out[f] = sigmaFace(__dmul_rn(mu, magSf[f]), s0[f], s1[f], s2[f], d0 ? d : nullptr, a0[f], a1[f], a2[f]);
}
__global__ void sigmaPatchKernel(int nB, scalar mu, const label* __restrict__ isEmpty,
                                 const label* __restrict__ gface, const scalar* __restrict__ magSf,
                                 const scalar* __restrict__ s0, const scalar* __restrict__ s1,
                                 const scalar* __restrict__ s2, const scalar* __restrict__ d0,
                                 const scalar* __restrict__ d1, const scalar* __restrict__ d2,
                                 const scalar* __restrict__ a0, const scalar* __restrict__ a1,
                                 const scalar* __restrict__ a2, scalar* __restrict__ out)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    if (isEmpty[k]) { out[k] = 0; return; }
    scalar d[3];
    if (d0) { d[0] = d0[k]; d[1] = d1[k]; d[2] = d2[k]; }
    out[k] = sigmaFace(__dmul_rn(mu, magSf[gface[k]]), s0[k], s1[k], s2[k], d0 ? d : nullptr, a0[k], a1[k], a2[k]);
}

// e's patch coefficients for - fvm::laplacian(alphaEff, e), from T's patch fields (heThermo's heBoundaryTypes):
// a T face holding a value (fixedValue; inletOutlet on inflow) is fixedEnergy (mixedEnergy, valueFraction 1):
// gradientInternalCoeffs -deltaCoeffs, gradientBoundaryCoeffs deltaCoeffs*e_b; a zeroGradient T face (inletOutlet on
// outflow) is gradientEnergy (mixedEnergy, valueFraction 0) whose gradient Cpv*snGrad(Tw) + dc*(he(Tw) - he(Tw)) is 0;
// a transform T (symmetryPlane, symmetry, slip) keeps its type, and transformFvPatchField<scalar> has no gradient
// coefficients (transformFvPatchFields.C:53-58). The matrix's = -(laplacian's): pGn = -alphaEff*magSf.
__global__ void energyPatchCoeffsKernel(int nB, scalar negAlpha, const label* __restrict__ isEmpty,
                                        const int* __restrict__ Tconstraint, const label* __restrict__ Ttype,
                                        const scalar* __restrict__ magSf, const scalar* __restrict__ dc,
                                        const scalar* __restrict__ eB, scalar* __restrict__ iC,
                                        scalar* __restrict__ bC)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    if (isEmpty[k] || Tconstraint[k] || Ttype[k] != 1)
    {
        iC[k] = 0;
        bC[k] = 0;
        return;
    }
    const scalar pGn = negAlpha*magSf[k];
    iC[k] = pGn*(-dc[k]);
    bC[k] = -pGn*(dc[k]*eB[k]);
}

// the step's viscous fluxes into the domain through the patches, times deltaT (diagnostics; atomics). For a matrix
// M = ddt - laplacian the laplacian's patch flux into the cell is internalCoeffs_lap*psi_c - boundaryCoeffs_lap
// = bC_M - iC_M*psi_c.
__global__ void viscousInflowKernel(int nB, scalar deltaT, const label* __restrict__ isEmpty,
                                    const label* __restrict__ bCell, int v0, int v1, int v2,
                                    const scalar* __restrict__ iC0, const scalar* __restrict__ bC0,
                                    const scalar* __restrict__ iC1, const scalar* __restrict__ bC1,
                                    const scalar* __restrict__ iC2, const scalar* __restrict__ bC2,
                                    const scalar* __restrict__ Ux, const scalar* __restrict__ Uy,
                                    const scalar* __restrict__ Uz, const scalar* __restrict__ dT0,
                                    const scalar* __restrict__ dT1, const scalar* __restrict__ dT2,
                                    const scalar* __restrict__ sigmaB, const scalar* __restrict__ iCe,
                                    const scalar* __restrict__ bCe, const scalar* __restrict__ e,
                                    double* __restrict__ inflow)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB || isEmpty[k]) return;
    const label c = bCell[k];
    double m0 = 0, m1 = 0, m2 = 0;
    if (v0) m0 += bC0[k] - iC0[k]*Ux[c];
    if (v1) m1 += bC1[k] - iC1[k]*Uy[c];
    if (v2) m2 += bC2[k] - iC2[k]*Uz[c];
    if (dT0) { m0 += dT0[k]; m1 += dT1[k]; m2 += dT2[k]; }
    double en = bCe[k] - iCe[k]*e[c];
    if (sigmaB) en += sigmaB[k];
    atomicAdd(&inflow[0], deltaT*m0);
    atomicAdd(&inflow[1], deltaT*m1);
    atomicAdd(&inflow[2], deltaT*m2);
    atomicAdd(&inflow[3], deltaT*en);
}

// the identity's pieces: rho V (x - x_p) on the cells, rho V |x| (its scale), and (bC - iC x_c) (+ Sf & tauMC_b) on the
// non-empty patch faces
__global__ void identityCellsKernel(int n, const scalar* __restrict__ rho, const scalar* __restrict__ V,
                                    const scalar* __restrict__ x, const scalar* __restrict__ xp,
                                    const scalar* __restrict__ s0, const scalar* __restrict__ s1,
                                    const scalar* __restrict__ s2, scalar* __restrict__ d, scalar* __restrict__ m)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    const scalar rv = rho[c]*V[c];
    d[c] = rv*(x[c] - xp[c]);
    // the scale: rho V |U| for a momentum component (a component near zero is no scale), rho V |e| for energy
    m[c] = s1 ? rv*sqrt(s0[c]*s0[c] + s1[c]*s1[c] + s2[c]*s2[c]) : rv*fabs(s0[c]);
}
__global__ void identityPatchKernel(int nB, const label* __restrict__ isEmpty, const label* __restrict__ bCell,
                                    const scalar* __restrict__ iC, const scalar* __restrict__ bC,
                                    const scalar* __restrict__ x, const scalar* __restrict__ tauB,
                                    scalar* __restrict__ f)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    if (isEmpty[k]) { f[k] = 0; return; }
    scalar v = bC[k] - iC[k]*x[bCell[k]];
    if (tauB) v += tauB[k];
    f[k] = v;
}

} // namespace

void ViscousCorrectors::account(int k, const DeviceLduView& A, const DeviceBuffer<scalar>& x,
                                const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& rho,
                                const DeviceBuffer<scalar>& iC, const DeviceBuffer<scalar>& bC,
                                const DeviceBuffer<scalar>* tauB, scalar rDeltaT, const DeviceBuffer<scalar>* const* sc)
{
    deviceResidual(A, x, b, res_);
    const double R = deviceDot(res_, ones_);
    solveRes_[k] += R/rDeltaT;
    solveResL1_[k] += deviceSumMag(res_)/rDeltaT;
    idC_.resize(nC_);
    DeviceBuffer<scalar>& m = idM_;
    m.resize(nC_);
    identityCellsKernel<<<blocksFor(nC_), kTPB>>>(nC_, rho.data(), dm_.V.data(), x.data(), xp_[k].data(),
                                                  sc[0]->data(), sc[1] ? sc[1]->data() : nullptr,
                                                  sc[2] ? sc[2]->data() : nullptr, idC_.data(), m.data());
    cudaCheck(cudaGetLastError(), "identity cells");
    const double lhs = deviceDot(idC_, ones_);
    const double scale = deviceDot(m, ones_);
    double bnd = 0;
    if (nB_)
    {
        idB_.resize(nB_);
        identityPatchKernel<<<blocksFor(nB_), kTPB>>>(nB_, dm_.bndIsEmpty.data(), bCell_.data(), iC.data(), bC.data(),
                                                      x.data(), tauB ? tauB->data() : nullptr, idB_.data());
        cudaCheck(cudaGetLastError(), "identity patches");
        bnd = deviceDot(idB_, onesB_);
    }
    const double err = std::fabs(lhs - (bnd - R)/rDeltaT)/(scale > 0 ? scale : 1.0);
    if (err > identityMax_[k]) identityMax_[k] = err;
}

ViscousCorrectors::ViscousCorrectors(const Case& cs, const DeviceMesh& dm, int control)
    : cs_(cs), dm_(dm), control_(control), nC_(cs.nC), nIF_(cs.nIF), nB_(cs.nB)
{
    for (int k = 0; k < 3; ++k) valid_[k] = cs.sd.valid(k);
    mu_ = cs.gas.mu;
    alphaEff_ = gasAlphaEff(cs.gas);
    for (std::size_t k = 0; k < cs.bUTransform.size(); ++k)
        if (cs.bUTransform[k] == transformRotate)
            throw std::runtime_error("brae rhoCentralFoam: a wedge patch with mu > 0. wedgeFvPatchField's snGrad "
                                     "and gradient coefficients take cellT (wedgeFvPatchField.C); the viscous "
                                     "correctors are not ported on wedge patches.");
    // the corrected schemes' explicit parts are exactly zero where every nonOrthCorrectionVector is: then skipped
    {
        const std::vector<scalar> cx = dm.corrVecX.host(), cy = dm.corrVecY.host(), cz = dm.corrVecZ.host();
        for (std::size_t f = 0; f < cx.size() && !nonOrth_; ++f)
            nonOrth_ = (cx[f] != 0 || cy[f] != 0 || cz[f] != 0);
    }
    muC_.copyFrom(std::vector<scalar>(nC_, mu_));
    muB_.copyFrom(std::vector<scalar>(nB_, mu_));
    muF_.copyFrom(std::vector<scalar>(nIF_, mu_));
    alphaF_.copyFrom(std::vector<scalar>(nIF_, alphaEff_));
    negMuB_.copyFrom(std::vector<scalar>(nB_, -mu_));
    negAlphaB_.copyFrom(std::vector<scalar>(nB_, -alphaEff_));
    ones_.copyFrom(std::vector<scalar>(nC_, 1.0));
    onesB_.copyFrom(std::vector<scalar>(nB_, 1.0));
    // the two laplacians' matrices (gaussLaplacianScheme: gammaMagSf = interpolate(gamma)*magSf, upper =
    // nonOrthDeltaCoeffs*gammaMagSf, diag = negSumDiag), and the corrector matrices' off-diagonal -upper
    DeviceBuffer<scalar> gms, upper;
    gms.resize(nIF_);
    for (int pass = 0; pass < 2; ++pass)
    {
        const scalar g = pass ? alphaEff_ : mu_;
        if (nIF_)
        {
            scaleKernel<<<blocksFor(nIF_), kTPB>>>(nIF_, g, dm.magSf.data(), gms.data());
            cudaCheck(cudaGetLastError(), "gammaMagSf");
        }
        DeviceBuffer<scalar>& d = pass ? lapDiagE_ : lapDiagU_;
        DeviceBuffer<scalar>& nu = pass ? negUpperE_ : negUpperU_;
        deviceLaplacianGammaMagSf(dm, gms, d, upper, /*nonOrth=*/true);
        nu.resize(nIF_);
        if (nIF_)
        {
            negateKernel<<<blocksFor(nIF_), kTPB>>>(nIF_, upper.data(), nu.data());
            cudaCheck(cudaGetLastError(), "-upper");
        }
    }
    gradU_.resize(9*static_cast<std::size_t>(nC_));
    for (int k = 0; k < 3; ++k)
    {
        U0_[k].resize(nC_);
        src_[k].resize(nC_);
    }
    {
        std::vector<int> tc(nB_);
        for (label k = 0; k < nB_; ++k) tc[k] = cs.bTseedOwn[k] ? 1 : 0;
        Tconstraint_.copyFrom(tc);
    }
    uKind_.copyFrom(cs.bUTransform);
    uR_.copyFrom(cs.bUR);
    bCell_.copyFrom(cs.bCell);
    cudaCheck(cudaDeviceSynchronize(), "viscous setup");
}

void ViscousCorrectors::beginStep(const DeviceFields& F, const DeviceVectorBoundary& dbU)
{
    copyD2D(rho0_, F.rho);
    copyD2D(U0_[0], F.Ux);
    copyD2D(U0_[1], F.Uy);
    copyD2D(U0_[2], F.Uz);
    copyD2D(e0_, F.e);

    // :218-219, tauMC = muEff*dev2(T(fvc::grad(U))): Gauss linear, packed [(d*3 + i)*nC + c] = d_d U_i, the patch
    // faces by gaussGrad::correctBoundaryConditions on U's stored patch values
    const DeviceBuffer<scalar>* vol[3] = {&F.Ux, &F.Uy, &F.Uz};
    const DeviceBuffer<scalar>* bv[3] = {&F.UxB, &F.UyB, &F.UzB};
    deviceGaussGradFused(dm_, 3, vol, bv, gx_, gy_, gz_);
    for (int i = 0; i < 3; ++i)
    {
        copySlice(gradU_, 0*3 + i, nC_, gx_[i]);
        copySlice(gradU_, 1*3 + i, nC_, gy_[i]);
        copySlice(gradU_, 2*3 + i, nC_, gz_[i]);
    }
    deviceBoundaryGradU(dm_, dbU, F.Ux, F.Uy, F.Uz, gradU_, gradB_, bv);
    deviceScaledDev2T(nC_, gradU_, muC_, tauC_);
    deviceScaledDev2T(nB_, gradB_, muB_, tauB_);
    // fvc::div(tauMC) (Gauss linear: surfaceIntegrate(dotInterpolate(Sf, tauMC))) and sigmaDotU's Sf & tauMC_f: the
    // same face field, from the same tauMC
    deviceDotInterpolateSf(dm_, tauC_, tauB_, dTau_, dTauB_);
    for (int k = 0; k < 3; ++k) deviceDiv(dm_, dTau_[k], dTauB_[k], divTau_[k]);
}

void ViscousCorrectors::solveU(DeviceFields& F, DeviceVectorBoundary& dbU, scalar rDeltaT, StepReport& rep)
{
    DeviceBuffer<scalar>* U[3] = {&F.Ux, &F.Uy, &F.Uz};
    // symmetryPlane/symmetry/slip coefficients from U as it stands (basicSymmetry's gradientBoundaryCoeffs read the
    // patch-internal field when the matrix is built)
    deviceUpdateSymmetry(dbU, F.Ux, F.Uy, F.Uz);
    if (nonOrth_)   // the corrected scheme's explicit part needs grad(U_k) of this U
    {
        const DeviceBuffer<scalar>* vol[3] = {&F.Ux, &F.Uy, &F.Uz};
        const DeviceBuffer<scalar>* bv[3] = {&F.UxB, &F.UyB, &F.UzB};
        deviceGaussGradFused(dm_, 3, vol, bv, gx_, gy_, gz_);
    }
    const bool tau = (control_ != controlNoTauMC);
    GSFusedComponent comps[3];
    int solved[3], n = 0;
    for (int k = 0; k < 3; ++k)
    {
        if (!valid_[k]) continue;   // fvMatrixSolve.C: components an empty direction knocks out are not solved
        deviceEulerFvmDdtRho(rDeltaT, F.rho, rho0_, U0_[k], dm_.V, diagD_, src_[k]);
        deviceEulerFvcDdtRhoSub(rDeltaT, F.rho, *U[k], rho0_, U0_[k], dm_.V, src_[k]);
        const scalar* corr = nullptr;
        if (nonOrth_)
        {
            deviceLaplacianCorrFlux(dm_, muF_, gx_[k], gy_[k], gz_[k], ffc_);
            deviceFaceDivSource(dm_, ffc_, lapCorr_);
            corr = lapCorr_.data();
        }
        diagM_.resize(nC_);
        finishMatrixKernel<<<blocksFor(nC_), kTPB>>>(nC_, diagD_.data(), lapDiagU_.data(), corr, dm_.V.data(),
                                                     tau ? divTau_[k].data() : nullptr, diagM_.data(),
                                                     src_[k].data());
        cudaCheck(cudaGetLastError(), "U corrector matrix");
        deviceBCLaplacianCoeffsFace(dbU.comp[k], negMuB_, iC_[k], bC_[k]);
        deviceFold(dm_, diagM_, src_[k], iC_[k], bC_[k], diagC_[k], b_[k]);
        A_[k] = deviceLduView(dm_, diagC_[k], negUpperU_, negUpperU_);
        deviceNormFactorInto(A_[k], *U[k], b_[k], ones_, dnf_[k]);
        comps[n] = {&A_[k], &b_[k], U[k], 1.0, dnf_[k].data()};
        solved[n++] = k;
    }
    DeviceSolverPerf perf[3];
    const SmoothSolveControls& c = cs_.solveU;
    for (int i = 0; i < n; ++i) copyD2D(xp_[solved[i]], *U[solved[i]]);   // U before the solve, for the identity
    if (n) deviceSymGaussSeidelFused(n, comps, c.tolerance, c.relTol, c.maxIter, c.minIter, c.nSweeps, c.symmetric, perf);
    for (int i = 0; i < n; ++i)   // the run's unconserved remainder deltaT*sum(b - A x), unsigned, and the identity
    {
        const DeviceBuffer<scalar>* sc[3] = {&F.Ux, &F.Uy, &F.Uz};
        account(solved[i], A_[solved[i]], *U[solved[i]], b_[solved[i]], F.rho, iC_[solved[i]], bC_[solved[i]],
                tau ? &dTauB_[solved[i]] : nullptr, rDeltaT, sc);
    }
    for (int i = 0; i < n && rep.nSolves < 4; ++i)
    {
        SolveLine& l = rep.solves[rep.nSolves++];
        const char* nm[3] = {"Ux", "Uy", "Uz"};
        std::strncpy(l.field, nm[solved[i]], sizeof(l.field) - 1);
        l.initialResidual = perf[i].initialResidual;
        l.finalResidual = perf[i].finalResidual;
        l.nIterations = perf[i].nIterations;
    }
    // :243 rhoU = rho*U (the cells; the patches are the caller's, with U's)
    DeviceBuffer<scalar>* rU[3] = {&F.rhoUx, &F.rhoUy, &F.rhoUz};
    for (int k = 0; k < 3; ++k)
    {
        rhoTimesKernel<<<blocksFor(nC_), kTPB>>>(nC_, F.rho.data(), U[k]->data(), rU[k]->data());
        cudaCheck(cudaGetLastError(), "rhoU = rho*U");
    }
}

const DeviceBuffer<scalar>& ViscousCorrectors::divSigmaDotU(const DeviceFields& F, const DeviceVectorBoundary& dbU,
                                                            const DeviceFluxes& X)
{
    const DeviceBuffer<scalar>* U[3] = {&F.Ux, &F.Uy, &F.Uz};
    if (nonOrth_)   // correctedSnGrad's explicit part needs grad(U_k) of the corrected U
    {
        const DeviceBuffer<scalar>* bv[3] = {&F.UxB, &F.UyB, &F.UzB};
        deviceGaussGradFused(dm_, 3, U, bv, gx_, gy_, gz_);
    }
    for (int k = 0; k < 3; ++k)
        deviceSnGradInternal(dm_, *U[k], snG_[k], /*nonOrth=*/true, nonOrth_ ? &gx_[k] : nullptr,
                             nonOrth_ ? &gy_[k] : nullptr, nonOrth_ ? &gz_[k] : nullptr);
    for (int k = 0; k < 3; ++k) snGB_[k].resize(nB_);
    if (nB_)
    {
        snGradPatchKernel<<<blocksFor(nB_), kTPB>>>(nB_, dm_.bndIsEmpty.data(), uKind_.data(), uR_.data(),
                                                    bCell_.data(), dbU.comp[0].deltaCoeffs.data(),
                                                    dbU.comp[0].bcType.data(), dbU.comp[1].bcType.data(),
                                                    dbU.comp[2].bcType.data(), F.Ux.data(), F.Uy.data(),
                                                    F.Uz.data(), F.UxB.data(), F.UyB.data(), F.UzB.data(),
                                                    snGB_[0].data(), snGB_[1].data(), snGB_[2].data());
        cudaCheck(cudaGetLastError(), "snGrad(U) patches");
    }
    const bool tau = (control_ != controlNoTauMC);
    sigma_.resize(nIF_);
    sigmaB_.resize(nB_);
    if (nIF_)
    {
        sigmaInternalKernel<<<blocksFor(nIF_), kTPB>>>(nIF_, mu_, dm_.magSf.data(), snG_[0].data(), snG_[1].data(),
                                                       snG_[2].data(), tau ? dTau_[0].data() : nullptr,
                                                       tau ? dTau_[1].data() : nullptr, tau ? dTau_[2].data() : nullptr,
                                                       X.aUx.data(), X.aUy.data(), X.aUz.data(), sigma_.data());
        cudaCheck(cudaGetLastError(), "sigmaDotU");
    }
    if (nB_)
    {
        sigmaPatchKernel<<<blocksFor(nB_), kTPB>>>(nB_, mu_, dm_.bndIsEmpty.data(), dm_.bndGFace.data(),
                                                   dm_.magSf.data(), snGB_[0].data(), snGB_[1].data(),
                                                   snGB_[2].data(), tau ? dTauB_[0].data() : nullptr,
                                                   tau ? dTauB_[1].data() : nullptr, tau ? dTauB_[2].data() : nullptr,
                                                   X.aUxB.data(), X.aUyB.data(), X.aUzB.data(), sigmaB_.data());
        cudaCheck(cudaGetLastError(), "sigmaDotU patches");
    }
    deviceDiv(dm_, sigma_, sigmaB_, divSigma_);
    return divSigma_;
}

void ViscousCorrectors::solveE(DeviceFields& F, const DeviceBoundary& dbT, scalar rDeltaT, StepReport& rep)
{
    deviceEulerFvmDdtRho(rDeltaT, F.rho, rho0_, e0_, dm_.V, diagD_, srcE_);
    deviceEulerFvcDdtRhoSub(rDeltaT, F.rho, F.e, rho0_, e0_, dm_.V, srcE_);
    const scalar* corr = nullptr;
    if (nonOrth_)
    {
        const DeviceBuffer<scalar>* vol[1] = {&F.e};
        const DeviceBuffer<scalar>* bv[1] = {&F.eB};
        deviceGaussGradFused(dm_, 1, vol, bv, &ge_[0], &ge_[1], &ge_[2]);
        deviceLaplacianCorrFlux(dm_, alphaF_, ge_[0], ge_[1], ge_[2], ffc_);
        deviceFaceDivSource(dm_, ffc_, lapCorr_);
        corr = lapCorr_.data();
    }
    diagE_.resize(nC_);
    finishMatrixKernel<<<blocksFor(nC_), kTPB>>>(nC_, diagD_.data(), lapDiagE_.data(), corr, dm_.V.data(), nullptr,
                                                 diagE_.data(), srcE_.data());
    cudaCheck(cudaGetLastError(), "e corrector matrix");
    iCe_.resize(nB_);
    bCe_.resize(nB_);
    if (nB_)
    {
        energyPatchCoeffsKernel<<<blocksFor(nB_), kTPB>>>(nB_, -alphaEff_, dm_.bndIsEmpty.data(), Tconstraint_.data(),
                                                          dbT.bcType.data(), dbT.magSf.data(), dbT.deltaCoeffs.data(),
                                                          F.eB.data(), iCe_.data(), bCe_.data());
        cudaCheck(cudaGetLastError(), "e patch coefficients");
    }
    deviceFold(dm_, diagE_, srcE_, iCe_, bCe_, diagCE_, bE_);
    AE_ = deviceLduView(dm_, diagCE_, negUpperE_, negUpperE_);
    deviceNormFactorInto(AE_, F.e, bE_, ones_, dnfE_);
    GSFusedComponent comp{&AE_, &bE_, &F.e, 1.0, dnfE_.data()};
    DeviceSolverPerf perf;
    const SmoothSolveControls& c = cs_.solveE;
    copyD2D(xp_[3], F.e);
    deviceSymGaussSeidelFused(1, &comp, c.tolerance, c.relTol, c.maxIter, c.minIter, c.nSweeps, c.symmetric, &perf);
    {
        const DeviceBuffer<scalar>* sc[3] = {&F.e, nullptr, nullptr};
        account(3, AE_, F.e, bE_, F.rho, iCe_, bCe_, nullptr, rDeltaT, sc);
    }
    if (rep.nSolves < 4)
    {
        SolveLine& l = rep.solves[rep.nSolves++];
        std::strncpy(l.field, "e", sizeof(l.field) - 1);
        l.initialResidual = perf.initialResidual;
        l.finalResidual = perf.finalResidual;
        l.nIterations = perf.nIterations;
    }
}

void ViscousCorrectors::boundaryInflow(const DeviceFields& F, scalar deltaT, DeviceBuffer<double>& inflow)
{
    if (!nB_) return;
    const bool tau = (control_ != controlNoTauMC);
    const bool sig = (control_ != controlNoSigmaDotU);
    auto p = [&](int k, const DeviceBuffer<scalar>* a) { return valid_[k] ? a[k].data() : nullptr; };
    viscousInflowKernel<<<blocksFor(nB_), kTPB>>>(nB_, deltaT, dm_.bndIsEmpty.data(), bCell_.data(), valid_[0],
                                                  valid_[1], valid_[2], p(0, iC_), p(0, bC_), p(1, iC_), p(1, bC_),
                                                  p(2, iC_), p(2, bC_), F.Ux.data(), F.Uy.data(), F.Uz.data(),
                                                  tau ? dTauB_[0].data() : nullptr, tau ? dTauB_[1].data() : nullptr,
                                                  tau ? dTauB_[2].data() : nullptr, sig ? sigmaB_.data() : nullptr,
                                                  iCe_.data(), bCe_.data(), F.e.data(), inflow.data());
    cudaCheck(cudaGetLastError(), "viscous inflow");
}

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
