// cf GPU offload -- boundary contributions to the DISCRETISED matrix/fluxes: the valueInternal/valueBoundary
// coeffs of each BC category (fvPatchField), for the value, laplacian, div and matrix-flux operators. One thread
// per boundary face; the BC category (bcType) selects the formula. Split from device_boundary.cu (the per-iteration
// flow-BC value updates are in device_boundary_flow.cu). Shared internal decls: device_boundary.cuh.
#include "device_boundary.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
void bcValueKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const label* __restrict__ fc,
    const scalar* __restrict__ internal,
    const scalar* __restrict__ rgr,     // fixedGradient g (null/zero elsewhere)
    const scalar* __restrict__ dcv,
    scalar* __restrict__ value,
    const int* __restrict__ skipIf)     // device flag: when set, this launch is a no-op (the grad(U) memo hit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (skipIf && *skipIf) return;

    if (type[i] == 8)      return;                                                       // coupled (processor): the
                                                                                         // face value is the halo-
                                                                                         // interpolated one, injected
                                                                                         // by DeviceHalo::scatterBoundaryValues -- never derived from the local cell.
    else if (type[i] == 0) value[i] = internal[fc[i]] + (rgr ? rgr[i] / dcv[i] : scalar(0));   // zeroGradient, or
                                                                                         // fixedGradient when refGrad
                                                                                         // is non-zero: OF's
                                                                                         // patchInternalField + g/deltaCoeffs

    // mixed (Robin). OF evaluate(): lerp(patchInternal + refGrad/dc, refValue, vf)
    // = (1-vf)*(internal + refGrad/dc) + vf*refValue  (mixedFvPatchField.C:239-247). The refGrad term rides
    // INSIDE the (1-vf) branch, which is what makes mixedEnergy's refGrad = Cpv*Tw.refGrad() land correctly.
    else if (type[i] == 5)
        value[i] = (1.0 - vf[i]) * (internal[fc[i]] + (rgr ? rgr[i] / dcv[i] : scalar(0))) + vf[i] * ref[i];
    else                   value[i] = ref[i];                                            // fixedValue / calculated
}


// mixed-aware laplacian gradient weight: w = vf (fixedValue vf=1 -> gradIC=-dc; zeroGradient vf=0 -> 0). vf[i] is
// read ONLY for type==5 (the ternary short-circuits), so a non-mixed boundary with no valueFraction is safe.
__global__
void bcLaplacianKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ gammaCell,
    const label* __restrict__ fc,
    const scalar* __restrict__ rgr,     // fixedGradient g (null/zero elsewhere)
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar pG = gammaCell[fc[i]] * magSf[i];                        // pGamma = gammaf_b * |Sf|
    const scalar w = (type[i] == 1) ? 1.0 : ((type[i] == 5) ? vf[i] : 0.0);
    iC[i] = pG * (-w * dc[i]);
    // + the fixedGradient term: OF's gradientBoundaryCoeffs = g, and brae's bC is -pG times OF's, so a
    // prescribed gradient adds -pG*g. refGrad is 0 on every other BC, so this is a no-op there.
    // OF gradientBoundaryCoeffs = lerp(refGrad, dc*refValue, vf) = vf*refValue*dc + (1-vf)*refGrad
    // (mixedFvPatchField.C:302-310). The refGrad weight is (1-vf), NOT 1 -- it is only 1 for
    // fixedGradient, where vf = 0. Getting that wrong is invisible until a `mixed` patch carries a
    // non-zero refGradient, which is exactly the mixedEnergy case.
    const scalar wg = (type[i] == 5) ? (scalar(1) - vf[i]) : scalar(1);
    bC[i] = -pG * (w * ref[i] * dc[i] + wg * (rgr ? rgr[i] : scalar(0)));
}


// same as bcLaplacianKernel but gamma is given per BOUNDARY FACE (e.g. nuEff = nu + nutkWallFunction at walls).
__global__
void bcLaplacianFaceKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ gammaFace,
    const scalar* __restrict__ rgr,
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar pG = gammaFace[i] * magSf[i];
    const scalar w = (type[i] == 1) ? 1.0 : ((type[i] == 5) ? vf[i] : 0.0);
    const scalar wg = (type[i] == 5) ? (scalar(1) - vf[i]) : scalar(1);   // OF's lerp weight on refGrad
    iC[i] = pG * (-w * dc[i]);
    // Same fixedGradient source as bcLaplacianKernel. It has to be in BOTH: the energy equation is the
    // only caller of the face-diffusivity variant, and a heat-flux wall is precisely an energy BC -- so
    // adding the term to the cell-diffusivity kernel alone leaves the one case B5 exists for unheated.
    bC[i] = -pG * (w * ref[i] * dc[i] + wg * (rgr ? rgr[i] : scalar(0)));
}


__global__
void bcDivKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const scalar* __restrict__ rgr,     // fixedGradient g (null/zero elsewhere)
    const scalar* __restrict__ dcv,
    const scalar* __restrict__ phiB,
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // A COUPLED (processor) face contributes NOTHING as a boundary: its coupling is the interface off-diagonal
    // (deviceMomentumInterface), so both coeffs are zero. Without this the default branch below would give
    // vIC = 1 -> iC = phi, DOUBLE-COUNTING the interface diagonal. Mirrors host ProcessorFvPatchField, whose
    // valueInternalCoeffs/valueBoundaryCoeffs are overridden to zero for exactly this reason.
    if (type[i] == 8)
    {
        iC[i] = 0.0;
        bC[i] = 0.0;
        return;
    }
    // valueInternalCoeffs: fixedValue 0, zeroGradient/calculated 1, mixed 1-vf. valueBoundaryCoeffs: fixedValue ref,
    // mixed vf*ref, else 0.
    const scalar vIC = (type[i] == 1) ? 0.0 : ((type[i] == 5) ? (1.0 - vf[i]) : 1.0);
    // + fixedGradient: OF's valueBoundaryCoeffs = g/deltaCoeffs (valueInternalCoeffs stays 1, as for
    // zeroGradient). Zero elsewhere.
    // OF valueBoundaryCoeffs = lerp(refGrad/dc, refValue, vf) = vf*refValue + (1-vf)*refGrad/dc
    // (mixedFvPatchField.C:279-290) -- again the (1-vf) weight on the gradient part.
    const scalar wgv = (type[i] == 5) ? (scalar(1) - vf[i]) : scalar(1);
    const scalar vBC = ((type[i] == 1) ? ref[i] : ((type[i] == 5) ? vf[i] * ref[i] : 0.0))
                     + wgv * (rgr ? rgr[i] / dcv[i] : scalar(0));
    iC[i] = phiB[i] * vIC;
    bC[i] = -phiB[i] * vBC;
}


__global__
void bcMatrixFluxKernel(
    int n,
    const label* __restrict__ fc,
    const scalar* __restrict__ iC,
    const scalar* __restrict__ bC,
    const scalar* __restrict__ p,
    scalar* __restrict__ fluxB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fluxB[i] = iC[i] * p[fc[i]] - bC[i];
}
} // namespace


void deviceBCValue(const DeviceBoundary& db, const DeviceBuffer<scalar>& internal, DeviceBuffer<scalar>& value, const int* skipIf)
{
    value.resize(db.n);
    bcValueKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                          db.faceCell.data(), internal.data(),
                                          db.refGrad.size() ? db.refGrad.data() : nullptr,
                                          db.deltaCoeffs.data(), value.data(), skipIf);
    cudaCheck(cudaGetLastError(), "bcValue");
}


void deviceBCLaplacianCoeffsFace(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& gammaFace,
    DeviceBuffer<scalar>& iC,
    DeviceBuffer<scalar>& bC)
{
    iC.resize(db.n);
    bC.resize(db.n);
    bcLaplacianFaceKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                                  db.deltaCoeffs.data(), db.magSf.data(), gammaFace.data(),
                                                  db.refGrad.size() ? db.refGrad.data() : nullptr, iC.data(), bC.data());
    cudaCheck(cudaGetLastError(), "bcLaplacianFace");
}


void deviceBCLaplacianCoeffs(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& gammaCell,
    DeviceBuffer<scalar>& iC,
    DeviceBuffer<scalar>& bC)
{
    iC.resize(db.n);
    bC.resize(db.n);
    bcLaplacianKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                              db.deltaCoeffs.data(), db.magSf.data(), gammaCell.data(), db.faceCell.data(),
                                              db.refGrad.size() ? db.refGrad.data() : nullptr,
                                              iC.data(), bC.data());
    cudaCheck(cudaGetLastError(), "bcLaplacian");
}


void deviceBCDivCoeffs(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& phiB,
    DeviceBuffer<scalar>& iC,
    DeviceBuffer<scalar>& bC)
{
    iC.resize(db.n);
    bC.resize(db.n);
    bcDivKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                        db.refGrad.size() ? db.refGrad.data() : nullptr, db.deltaCoeffs.data(),
                                        phiB.data(), iC.data(), bC.data());
    cudaCheck(cudaGetLastError(), "bcDiv");
}


void deviceMatrixFluxBoundary(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& iC,
    const DeviceBuffer<scalar>& bC,
    const DeviceBuffer<scalar>& p,
    DeviceBuffer<scalar>& fluxB)
{
    fluxB.resize(db.n);
    bcMatrixFluxKernel<<<nBlocks(db.n), TPB>>>(db.n, db.faceCell.data(), iC.data(), bC.data(), p.data(), fluxB.data());
    cudaCheck(cudaGetLastError(), "bcMatrixFlux");
}

} // namespace brae
