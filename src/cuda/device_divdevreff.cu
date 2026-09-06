// cf GPU offload: explicit divDevReff stress source. Tensors packed component-major (i*3+j)*n + cell.
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"   // deviceCellLimitGradU (the named grad(U) scheme)
#include "stage_dump.cuh"   // sigma comparison against OF (ACMI trace)
#include "device_blas.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


// sigma = nuEff * dev2(transpose(gradU)) per cell. T(gradU)_ij = g[j][i]; dev2 subtracts (2/3)tr on diag.
__global__
void sigmaKernel(
    int n,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ nu,
    scalar* __restrict__ sigma)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    scalar g[9];
    for (int q = 0; q < 9; ++q)
        g[q] = gradU[q * n + c];

    const scalar tr = g[0] + g[4] + g[8];
    const scalar ne = nu[c];

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar T = g[j * 3 + i];   // transpose
            const scalar d = (i == j) ? (2.0 / 3.0) * tr : 0.0;
            sigma[(i * 3 + j) * n + c] = ne * (T - d);
        }
}


// One component's boundary coefficients, as the patch classes carry them. Passed by value so the kernel
// can ask each component separately: a directionMixed patch (pressureInletOutletVelocity) fixes one
// direction and leaves another free, so its valueFraction is per COMPONENT.
struct BndSnGradView
{
    const label*  type;
    const scalar* vf;
    const scalar* ref;
    const scalar* rgr;   // may be null: no fixedGradient face on this component
};

// boundary gradient: gradB = gradC + n (x) (snGrad - n & gradC), OF gaussGrad::correctBoundaryConditions.
//
// snGrad IS THE PATCH'S OWN, not (U_b - U_cell)*deltaCoeffs. That inline formula is only
// fvPatchField::snGrad -- the BASE class's -- and OF overrides it on the classes that matter here:
// zeroGradient returns exactly zero (its .H), fixedGradient the prescribed gradient (its .H), and mixed
// lerp(refGrad, (refValue - pif)*deltaCoeffs, valueFraction) using the CURRENT valueFraction while
// value() still carries the blend of the previous one. The host reference made the same mistake and it
// cost naca0012 3.0e-07 on div(dev2(T(grad(U)))) -- see fv_patch_field.cuh's snGrad (queue item 25).
//
// The device's type codes carry the class: 0 = extrapolated, which is zeroGradient when refGrad is zero
// and fixedGradient when it is not -- so snGrad = refGrad serves both, exactly as OF's two overrides do.
// 1/2 (fixedValue/calculated) keep the base formula, which is what OF uses for them. 5 is mixed. 8 is
// COUPLED, and OF's correction skips coupled patches outright (`if (!coupled())`), so the face keeps the
// extrapolated cell gradient.
__global__
void gradBKernel(
    int nB,
    const label* __restrict__ fc,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ gradU,
    int nC,
    const scalar* __restrict__ uxb,
    const scalar* __restrict__ uyb,
    const scalar* __restrict__ uzb,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ dc,
    BndSnGradView bx,
    BndSnGradView by,
    BndSnGradView bz,
    scalar* __restrict__ gradB)
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= nB) return;

    const int c = fc[bi], f = bndGFace[bi];

    scalar nx = Sfx[f], ny = Sfy[f], nz = Sfz[f];
    const scalar mag = sqrt(nx * nx + ny * ny + nz * nz);
    nx /= mag;
    ny /= mag;
    nz /= mag;

    scalar gc[9];
    for (int q = 0; q < 9; ++q)
        gc[q] = gradU[q * nC + c];

    // A coupled face keeps the extrapolated cell gradient: OF's correction runs only on !coupled().
    if (bx.type[bi] == 8)
    {
        for (int q = 0; q < 9; ++q) gradB[q * nB + bi] = gc[q];
        return;
    }

    const scalar nv[3] = { nx, ny, nz };
    const scalar ub[3] = { uxb[bi], uyb[bi], uzb[bi] };
    const scalar uc[3] = { Ux[c], Uy[c], Uz[c] };
    const BndSnGradView bv[3] = { bx, by, bz };
    scalar sn[3];
    for (int k = 0; k < 3; ++k)
    {
        const label t = bv[k].type[bi];
        const scalar rg = bv[k].rgr ? bv[k].rgr[bi] : scalar(0);
        if (t == 1 || t == 2)   sn[k] = (ub[k] - uc[k]) * dc[bi];                          // fvPatchField
        else if (t == 5)        sn[k] = bv[k].vf[bi] * (bv[k].ref[bi] - uc[k]) * dc[bi]     // mixed
                                      + (scalar(1) - bv[k].vf[bi]) * rg;
        else                    sn[k] = rg;                       // zeroGradient (0) / fixedGradient
    }

    // (n & gradC)_j = sum_i n_i gc[i][j]
    scalar ngc[3];
    for (int j = 0; j < 3; ++j)
        ngc[j] = nv[0]*gc[0*3+j] + nv[1]*gc[1*3+j] + nv[2]*gc[2*3+j];

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            gradB[(i*3+j)*nB + bi] = gc[i*3+j] + nv[i] * (sn[j] - ngc[j]);
}

inline BndSnGradView snGradView(const DeviceBoundary& db)
{
    BndSnGradView v{};
    v.type = db.bcType.data();
    v.vf   = db.valueFraction.data();
    v.ref  = db.refValue.data();
    v.rgr  = db.refGrad.size() ? db.refGrad.data() : nullptr;
    return v;
}


// tensor divergence gathered per cell: divSig_j = (1/V)[ sum_faces (Sf & sigma_face)_j ].
__global__
void tensorDivKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndGFace,
    const label* __restrict__ bndIsEmpty,
    const scalar* __restrict__ sigmaC,
    const scalar* __restrict__ sigmaB,
    int nB,
    const scalar* __restrict__ V,
    scalar* __restrict__ dX,
    scalar* __restrict__ dY,
    scalar* __restrict__ dZ)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar d[3] = { 0, 0, 0 };

    // internal faces owned by c (+): sigma_face = w*own + (1-w)*nei
    for (int fi = ownerStart[c]; fi < ownerStart[c + 1]; ++fi)
    {
        const int o = own[fi], n2 = nei[fi];
        const scalar wf = w[fi];
        const scalar sx = Sfx[fi], sy = Sfy[fi], sz = Sfz[fi];
        for (int j = 0; j < 3; ++j)
        {
            const scalar s0 = wf*sigmaC[(0*3+j)*nC+o] + (1.0-wf)*sigmaC[(0*3+j)*nC+n2];
            const scalar s1 = wf*sigmaC[(1*3+j)*nC+o] + (1.0-wf)*sigmaC[(1*3+j)*nC+n2];
            const scalar s2 = wf*sigmaC[(2*3+j)*nC+o] + (1.0-wf)*sigmaC[(2*3+j)*nC+n2];
            d[j] += sx*s0 + sy*s1 + sz*s2;
        }
    }

    // internal faces neighbouring c (-)
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int fi = losort[k];
        const int o = own[fi], n2 = nei[fi];
        const scalar wf = w[fi];
        const scalar sx = Sfx[fi], sy = Sfy[fi], sz = Sfz[fi];
        for (int j = 0; j < 3; ++j)
        {
            const scalar s0 = wf*sigmaC[(0*3+j)*nC+o] + (1.0-wf)*sigmaC[(0*3+j)*nC+n2];
            const scalar s1 = wf*sigmaC[(1*3+j)*nC+o] + (1.0-wf)*sigmaC[(1*3+j)*nC+n2];
            const scalar s2 = wf*sigmaC[(2*3+j)*nC+o] + (1.0-wf)*sigmaC[(2*3+j)*nC+n2];
            d[j] -= sx*s0 + sy*s1 + sz*s2;
        }
    }

    // boundary faces (+): sigma_b at the boundary face (empty patches excluded, as in CPU fvc::div)
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int bi = bndPerm[k];
        if (bndIsEmpty[bi]) continue;
        const int f = bndGFace[bi];
        const scalar sx = Sfx[f], sy = Sfy[f], sz = Sfz[f];
        for (int j = 0; j < 3; ++j)
            d[j] += sx*sigmaB[(0*3+j)*nB+bi] + sy*sigmaB[(1*3+j)*nB+bi] + sz*sigmaB[(2*3+j)*nB+bi];
    }

    // = V*fvc::div (the /V and *V cancel -> raw sum)
    dX[c] = d[0];
    dY[c] = d[1];
    dZ[c] = d[2];
}
} // namespace


// The two pieces of the stress path the MAXWELL model needs, exported so it does not have to
// re-implement them: the gaussGrad-corrected BOUNDARY gradient, and the tensor divergence itself
// (V*fvc::div(T), the source convention brae's momentum assembly uses). Both are the same kernels
// divDevReff runs; sharing them is what keeps `div(nuM*grad(U))` consistent with
// `div(nu*dev2(T(grad(U))))` down to the boundary treatment.
void deviceBoundaryGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                         const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                         const DeviceBuffer<scalar>& gradU, DeviceBuffer<scalar>& gradB)
{
    const int nC = dm.nCells, nB = dm.nBndFaces;
    DeviceBuffer<scalar> uxb, uyb, uzb;
    deviceBCValue(dbU.comp[0], Ux, uxb);
    deviceBCValue(dbU.comp[1], Uy, uyb);
    deviceBCValue(dbU.comp[2], Uz, uzb);
    gradB.resize(static_cast<std::size_t>(9) * nB);
    gradBKernel<<<nBlocks(nB), TPB>>>(nB, dm.bndCell.data(), dm.bndGFace.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                      gradU.data(), nC, uxb.data(), uyb.data(), uzb.data(), Ux.data(), Uy.data(), Uz.data(),
                                      dbU.comp[0].deltaCoeffs.data(),
                                      snGradView(dbU.comp[0]), snGradView(dbU.comp[1]), snGradView(dbU.comp[2]),
                                      gradB.data());
    cudaCheck(cudaGetLastError(), "boundaryGradU");
}

void deviceTensorDivSource(const DeviceMesh& dm,
                           const DeviceBuffer<scalar>& Tcell, const DeviceBuffer<scalar>& Tbnd,
                           DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ,
                           const DeviceCyclic* cyc, const DeviceAMI* ami)
{
    const int nC = dm.nCells, nB = dm.nBndFaces;
    srcX.resize(nC);
    srcY.resize(nC);
    srcZ.resize(nC);
    tensorDivKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
                                          dm.nei.data(), dm.w.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                          dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(), dm.bndIsEmpty.data(),
                                          Tcell.data(), Tbnd.data(), nB, dm.V.data(), srcX.data(), srcY.data(), srcZ.data());
    cudaCheck(cudaGetLastError(), "tensorDivSource");
    if (cyc) interfaceAddTensorDiv(*cyc, Tcell, nC, srcX, srcY, srcZ);
    if (ami) interfaceAddTensorDiv(*ami, Tcell, nC, srcX, srcY, srcZ);
}


void deviceDivDevReff(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& nuCell,
    const DeviceBuffer<scalar>& nuBnd,
    DeviceBuffer<scalar>& srcX,
    DeviceBuffer<scalar>& srcY,
    DeviceBuffer<scalar>& srcZ,
    const DeviceCyclic* cyc,
    const DeviceAMI* ami,
    const DeviceProcStress* proc,
    const DeviceBuffer<scalar>* const* UbStored,
    scalar gradULimitK)
{
    const int nC = dm.nCells, nB = dm.nBndFaces;
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };

    // gradU (packed 9*nC): row i = gaussGrad(U_i).
    DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * nC);
    DeviceBuffer<scalar> uxb, uyb, uzb;   // boundary values (reused for snGrad)
    DeviceBuffer<scalar>* ub[3] = { &uxb, &uyb, &uzb };

    // rotational AMI: the gradU neighbour is the ROTATED AMI-interpolated U (forwardT.interp), precompute it.
    DeviceBuffer<scalar> amiUNx, amiUNy, amiUNz;
    if (ami && ami->rotational)
        deviceAmiInterpolateVec(*ami, Ux, Uy, Uz, amiUNx, amiUNy, amiUNz);
    DeviceBuffer<scalar>* amiUN[3] = { &amiUNx, &amiUNy, &amiUNz };

    // gaussGrad(U_i) = (dUi/dx, dUi/dy, dUi/dz) = COLUMN i of G (OF convention G_ij = dU_j/dx_i, via outer(Sf,U)).
    for (int i = 0; i < 3; ++i)
    {
        // U's boundary as OF's fvc::grad(U) reads it: the STORED value when the caller keeps one, and a
        // re-derivation only when it does not. The two agree only while the caller evaluates U's boundary
        // before every assembly, which OpenFOAM does not -- see the header (queue items 25, 30).
        DeviceBuffer<scalar> bval;
        if (UbStored && UbStored[i] && UbStored[i]->size() == static_cast<std::size_t>(nB))
            deviceCopy(bval, *UbStored[i]);
        else
            deviceBCValue(dbU.comp[i], *Uc[i], bval);
        if (proc)   // processor faces are bcType 8: deviceBCValue leaves them, the halo supplies the value
        {
            proc->halo->exchange(Uc[i]->data());
            proc->halo->scatterBoundaryValues(Uc[i]->data(), *proc->weights, *proc->procStart, bval.data());
            proc->halo->waitExchange();   // recv buffer is reused by the next component -- see device_halo.cuh
        }
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, *Uc[i], bval, gx, gy, gz);

        if (cyc)   // + cyclic faces in gradU (periodic consistency)
        {
            if (cyc->rotational) deviceCyclicAddGradRot(*cyc, Ux, Uy, Uz, i, dm.V, gx, gy, gz);   // neighbour rotated
            else interfaceAddGrad(*cyc, *Uc[i], dm.V, gx, gy, gz);
        }
        if (ami)
        {
            if (ami->rotational) deviceAmiAddGradRot(*ami, *Uc[i], *amiUN[i], dm.V, gx, gy, gz);   // neighbour rotated
            else interfaceAddGrad(*ami, *Uc[i], dm.V, gx, gy, gz);
        }

        cudaCheck(cudaMemcpy(gradU.data() + (0*3+i)*nC, gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice), "ddr g");
        cudaCheck(cudaMemcpy(gradU.data() + (1*3+i)*nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice), "ddr g");
        cudaCheck(cudaMemcpy(gradU.data() + (2*3+i)*nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice), "ddr g");
        *ub[i] = std::move(bval);
    }

    // ...then LIMIT it, if the case named a limited grad(U). OF applies cellLimitedGrad to the base
    // Gauss gradient AFTER the coupled-patch contributions are in, which is exactly here.
    if (gradULimitK > scalar(0)) deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradU, gradULimitK, cyc, ami);

    // sigma cell
    DeviceBuffer<scalar> sigmaC(static_cast<std::size_t>(9) * nC);
    sigmaKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), nuCell.data(), sigmaC.data());
    cudaCheck(cudaGetLastError(), "ddr sigmaC");

    // boundary gradient + sigma boundary
    DeviceBuffer<scalar> gradB(static_cast<std::size_t>(9) * nB);
    gradBKernel<<<nBlocks(nB), TPB>>>(nB, dm.bndCell.data(), dm.bndGFace.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                      gradU.data(), nC, uxb.data(), uyb.data(), uzb.data(), Ux.data(), Uy.data(), Uz.data(),
                                      /* bnd deltaCoeffs */ dbU.comp[0].deltaCoeffs.data(),
                                      snGradView(dbU.comp[0]), snGradView(dbU.comp[1]), snGradView(dbU.comp[2]),
                                      gradB.data());
    cudaCheck(cudaGetLastError(), "ddr gradB");
    DeviceBuffer<scalar> sigmaB(static_cast<std::size_t>(9) * nB);
    sigmaKernel<<<nBlocks(nB), TPB>>>(nB, gradB.data(), nuBnd.data(), sigmaB.data());
    cudaCheck(cudaGetLastError(), "ddr sigmaB");

    // Processor faces: OVERWRITE the sigmaB just computed from dev2(T(gradB)). A cut face is interior to the
    // global mesh, so its stress is the INTERPOLATED cell sigma (host parallelDivDevReff reads it from
    // distributeFromCells<tensor>(sigC, P)), not a boundary-gradient stress. Doing this per tensor component
    // keeps the exchange on the existing scalar halo. The zero is required: scatterBoundaryValues atomicAdds.
    if (proc)
    {
        for (int q = 0; q < 9; ++q)
        {
            for (int i = 0; i < proc->halo->nInterfaces(); ++i)
            {
                const std::size_t n = proc->halo->size(i);
                if (n == 0) continue;
                cudaCheck(
                    cudaMemsetAsync(
                        sigmaB.data() + static_cast<std::size_t>(q) * nB + (*proc->procStart)[i],
                        0,
                        n * sizeof(scalar),
                        cudaStreamPerThread),
                    "ddr sigmaB proc zero");
            }
            const scalar* sigC_q = sigmaC.data() + static_cast<std::size_t>(q) * nC;
            proc->halo->exchange(sigC_q);
            proc->halo->scatterBoundaryValues(sigC_q, *proc->weights, *proc->procStart,
                                              sigmaB.data() + static_cast<std::size_t>(q) * nB);
            proc->halo->waitExchange();   // the next component reuses the recv buffer
        }
    }

    if (stageDumpActive() && stageDumpFirstOnly("ddrSigma"))
    {   // sigma = nuEff*dev2(T(grad(U))) per cell, the input to the tensor divergence. Packed 9*nC,
        // [q*nC + c] with q = 3*row + col, so it can be compared against OF's volTensorField directly.
        stageDump("stage_ddr_sigma", sigmaC);
    }
    // tensor divergence (= V*fvc::div)
    srcX.resize(nC);
    srcY.resize(nC);
    srcZ.resize(nC);
    tensorDivKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
                                          dm.nei.data(), dm.w.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                          dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(), dm.bndIsEmpty.data(),
                                          sigmaC.data(), sigmaB.data(), nB, dm.V.data(), srcX.data(), srcY.data(), srcZ.data());
    cudaCheck(cudaGetLastError(), "ddr tensorDiv");
    if (cyc) interfaceAddTensorDiv(*cyc, sigmaC, nC, srcX, srcY, srcZ);   // + cyclic-face stress flux (V*fvc::div)
    if (ami) interfaceAddTensorDiv(*ami, sigmaC, nC, srcX, srcY, srcZ);      // + AMI-face stress flux
}

} // namespace brae
