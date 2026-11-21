// compressibleRAS.cu -- OpenFOAM's compressible RAS kOmegaSST layers on the device (see the header).
#include "compressibleRAS.cuh"

#include "device_blas.cuh"
#include "device_komega_sst.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace brae {
namespace gpu {
namespace compressibleRAS {

namespace {

constexpr int kTPB = 256;
inline int blocksFor(int n) { return (n + kTPB - 1)/kTPB; }

// CompressibleTurbulenceModel::muEff() = mut() + mu(), mut() = rho*nut: two field operations, each rounded
__global__ void muEffKernel(int n, const scalar* __restrict__ rho, const scalar* __restrict__ nut,
                            const scalar* __restrict__ mu, scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __dmul_rn(rho[i], nut[i]) + mu[i];
}

// nu() = mu/rho
__global__ void nuKernel(int n, const scalar* __restrict__ mu, const scalar* __restrict__ rho, scalar* __restrict__ nu)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) nu[i] = mu[i]/rho[i];
}

// a constraint patch's evaluate on a scalar: the face cell's value (basicSymmetry: (pif + transform(pif))/2 = pif;
// wedge: transform(pif) = pif)
__global__ void constraintKernel(int nB, const label* __restrict__ mask, const label* __restrict__ cell,
                                 const scalar* __restrict__ nut, const scalar* __restrict__ alphat,
                                 scalar* __restrict__ nutB, scalar* __restrict__ alphatB)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB || !mask[k]) return;
    nutB[k] = nut[cell[k]];
    alphatB[k] = alphat[cell[k]];
}

} // namespace

void KOmegaSST::constrainPatches()
{
    if (!nB_) return;
    constraintKernel<<<blocksFor(nB_), kTPB>>>(nB_, constraint_.data(), bCell_.data(), nut_.data(), alphat_.data(),
                                               nutB_.data(), alphatB_.data());
    cudaCheck(cudaGetLastError(), "compressible RAS: constraint patches");
}

KOmegaSST::KOmegaSST(const brae::compressibleRAS::Case& c, const PrimitiveMesh& m, const FvGeometry& g,
                     const std::vector<FvPatch>& patches, const GeometricField<vector>& U, const DeviceMesh& dm)
    : c_(c), dm_(dm), nC_(static_cast<int>(m.nCells())), nB_(static_cast<int>(c.kB.size()))
{
    k_.copyFrom(c.k.internal);
    omega_.copyFrom(c.omega.internal);
    nut_.copyFrom(c.nut.internal);
    nutB_.copyFrom(c.nutB);
    alphat_.copyFrom(c.alphat.internal);
    alphatB_.copyFrom(c.alphatB);
    // the closure's boundaries take the stored-value seed (DeviceBoundary::ioStored): an inletOutlet k or omega patch
    // holds its file `value` until its first flux switch, as OpenFOAM's construction leaves it
    dbK_ = buildDeviceBoundary(c.k, patches, g, /*storedIoSeed=*/true);
    dbOmega_ = buildDeviceBoundary(c.omega, patches, g, /*storedIoSeed=*/true);
    wall_ = buildDeviceWallData(m, g, patches, U, c.wfPatch);
    wfMask_.copyFrom(c.wfMask);
    f1One_.copyFrom(c.f1One);
    nutCalc_.copyFrom(c.nutCalc);
    nutKind_.copyFrom(c.nutKind);
    alphatMask_.copyFrom(c.alphatMask);
    yCell_.copyFrom(c.yCell);
    yBnd_.copyFrom(c.yBnd);
    nutCmu25_.copyFrom(c.nutCmu25);
    nutKappa_.copyFrom(c.nutKappa);
    nutE_.copyFrom(c.nutE);
    nutYplLam_.copyFrom(c.nutYplLam);
    alphatPrt_.copyFrom(c.alphatPrt);
    constraint_.copyFrom(c.constraintFace);
    bCell_.copyFrom(c.bCell);
    // wall-face order -> boundary face, as buildDeviceWallData decided it
    if (wall_.wfBnd.size()) wfBnd_.copyFrom(wall_.wfBnd.host());

    // the parts of the closure's input fixed for the run
    in_.co = c.co;
    in_.Prt = c.Prt;
    in_.boundedK = in_.boundedOmega = c.kDiv.bounded;
    in_.limitedLinear = c.kDiv.limitedLinear;
    in_.limiterCoeff = c.kDiv.limiterCoeff;
    in_.linearUpwind = c.kDiv.linearUpwind;
    // the gradient each equation's linearUpwind names, its cellLimited coefficient (compressibleRAS_cpp.cu divScheme):
    // left at 0 the correction ran on an unlimited Gauss gradient whatever the entry said
    in_.luGradK = c.kDiv.luGradK;
    omegaDiv_.bounded = c.omegaDiv.bounded;
    omegaDiv_.limitedLinear = c.omegaDiv.limitedLinear;
    omegaDiv_.limiterCoeff = c.omegaDiv.limiterCoeff;
    omegaDiv_.linearUpwind = c.omegaDiv.linearUpwind;
    omegaDiv_.luGradK = c.omegaDiv.luGradK;
    // kOmegaSSTBase::correct's omegaEqn.relax() and kEqn.relax() (kOmegaSSTBase.C:590, :616), never read before: the
    // closure's solveScalarEqn applies fvMatrix::relax(alpha) where the case names a factor (Case::relaxK)
    in_.relaxEquationK = c.relaxK;
    in_.relaxK = c.relaxKFactor;
    in_.relaxEquationOmega = c.relaxOmega;
    in_.relaxOmega = c.relaxOmegaFactor;
    in_.omegaDiv = &omegaDiv_;
    in_.omegaGrad = &omegaGrad_;
    in_.correctedLaplacian = c.correctedLaplacian;
    in_.tol = c.kSolve.tol;
    in_.relTol = c.kSolve.relTol;
    in_.maxIter = c.kSolve.maxIter;
    in_.minIter = c.kSolve.minIter;
    in_.gsK = in_.gsOmega = true;
    in_.gsSymmetric = c.kSolve.symmetric;
    in_.nSweepsKE = c.kSolve.nSweeps;
    svOmega_.tol = c.omegaSolve.tol;
    svOmega_.relTol = c.omegaSolve.relTol;
    svOmega_.maxIter = c.omegaSolve.maxIter;
    svOmega_.minIter = c.omegaSolve.minIter;
    svOmega_.nSweeps = c.omegaSolve.nSweeps;
    svOmega_.gsSymmetric = c.omegaSolve.symmetric;
    in_.omegaSolve = &svOmega_;
    in_.wfBndMask = &wfMask_;
    in_.wallYBndFace = &yBnd_;
    in_.f1OneMask = &f1One_;
    in_.nutCalcMask = &nutCalc_;
    in_.nutWfKindBnd = &nutKind_;
    in_.nutWfCmu25Bnd = &nutCmu25_;
    in_.nutWfKappaBnd = &nutKappa_;
    in_.nutWfEBnd = &nutE_;
    in_.nutWfYplLamBnd = &nutYplLam_;
    in_.alphatWallMask = &alphatMask_;
    in_.alphatPrtFace = &alphatPrt_;
    in_.yCell = &yCell_;
}

void KOmegaSST::fill(const Flow& f)
{
    in_.phiInt = f.phiInt;
    in_.phiBnd = f.phiBnd;
    // phi() = phi/fvc::interpolate(rho): the internal faces by the linear interpolate, the patches by rho's own patch
    // values (the interpolate's boundary is the field's)
    // (validate() has no flux yet: correctNut reads none)
    if (f.phiInt && dm_.nInternalFaces)
    {
        deviceInterpolate(dm_, *f.rho, rhoF_);
        if (f.controlPhiUndivided) deviceCopy(phiByRhoI_, *f.phiInt);
        else deviceDivide(phiByRhoI_, *f.phiInt, rhoF_);
    }
    if (f.phiBnd && nB_)
    {
        if (f.controlPhiUndivided) deviceCopy(phiByRhoB_, *f.phiBnd);
        else deviceDivide(phiByRhoB_, *f.phiBnd, *f.rhoB);
    }
    in_.phiByRhoInt = &phiByRhoI_;
    in_.phiByRhoBnd = &phiByRhoB_;
    in_.rhoCell = f.rho;
    in_.rhoBndFace = f.rhoB;
    in_.rhoOldCell = f.rhoOld;
    in_.rDeltaT = f.rDeltaT;
    in_.rDeltaTCells = f.rDeltaTCells;
    // nu() = mu/rho, cells, patch faces, wall faces
    nuC_.resize(nC_);
    nuKernel<<<blocksFor(nC_), kTPB>>>(nC_, f.mu->data(), f.rho->data(), nuC_.data());
    if (nB_)
    {
        nuB_.resize(nB_);
        nuKernel<<<blocksFor(nB_), kTPB>>>(nB_, f.muB->data(), f.rhoB->data(), nuB_.data());
    }
    cudaCheck(cudaGetLastError(), "compressible RAS: nu");
    in_.nuCell = &nuC_;
    in_.nuBndFace = &nuB_;
    const int nW = static_cast<int>(wfBnd_.size());
    if (nW > 0)
    {
        deviceGatherWallNu(wfBnd_, nuB_, nuW_);
        in_.nuWallFace = &nuW_;
    }
    // the wall nut as the call is entered (the wall functions' G0 reads it; correct() overwrites nutB)
    if (nB_) deviceCopy(nutBIn_, nutB_);
    in_.nutBndFace = &nutBIn_;
    if (nW > 0)
    {
        deviceGatherWallNu(wfBnd_, nutBIn_, nutWIn_);
        in_.nutWallFace = &nutWIn_;
    }
    in_.Ux = f.Ux;
    in_.Uy = f.Uy;
    in_.Uz = f.Uz;
}

void KOmegaSST::validate(const Flow& f, const DeviceVectorBoundary& dbU)
{
    fill(f);
    // correctNut() = correctNut(2*magSqr(symm(fvc::grad(U)))) (kOmegaSSTBase.C)
    DeviceBuffer<scalar> gradU, S2;
    deviceGradU(dm_, dbU, *f.Ux, *f.Uy, *f.Uz, gradU);
    deviceS2(gradU, nC_, S2);
    gpu::kOmegaSSTRAS::correctNut(k_, omega_, nut_, nutB_, &alphat_, &alphatB_, S2, gradU, dm_, dbU, dbK_, dbOmega_, in_);
    constrainPatches();
}

void KOmegaSST::muEff(const DeviceBuffer<scalar>& rho, const DeviceBuffer<scalar>& rhoB, const DeviceBuffer<scalar>& mu,
                      const DeviceBuffer<scalar>& muB, DeviceBuffer<scalar>& muEffC, DeviceBuffer<scalar>& muEffB) const
{
    muEffC.resize(nC_);
    muEffKernel<<<blocksFor(nC_), kTPB>>>(nC_, rho.data(), nut_.data(), mu.data(), muEffC.data());
    if (nB_)
    {
        muEffB.resize(nB_);
        muEffKernel<<<blocksFor(nB_), kTPB>>>(nB_, rhoB.data(), nutB_.data(), muB.data(), muEffB.data());
    }
    cudaCheck(cudaGetLastError(), "compressible RAS: muEff");
}

void KOmegaSST::correct(const Flow& f, const DeviceVectorBoundary& dbU)
{
    fill(f);
    gpu::kOmegaSSTRAS::correct(k_, omega_, nut_, nutB_, &alphat_, &alphatB_, res_, dm_, dbU, dbK_, dbOmega_, wall_, in_);
    constrainPatches();
}

void KOmegaSST::kB(DeviceBuffer<scalar>& out) const { deviceBCValue(dbK_, k_, out); }
void KOmegaSST::omegaB(DeviceBuffer<scalar>& out) const { deviceBCValue(dbOmega_, omega_, out); }

} // namespace compressibleRAS
} // namespace gpu
} // namespace brae
