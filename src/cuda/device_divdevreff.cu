// cf GPU offload: explicit divDevReff stress source. Tensors packed component-major (i*3+j)*n + cell.
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"   // deviceCellLimitGradU (the named grad(U) scheme)
#include "stage_dump.cuh"   // sigma comparison against OF (ACMI trace)
#include "device_blas.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


// sigma = nuEff * dev2(transpose(gradU)) per cell. T(gradU)_ij = g[j][i]; dev2 subtracts (2/3)tr on diag.
__device__
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
__device__
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
__device__
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
                         const DeviceBuffer<scalar>& gradU, DeviceBuffer<scalar>& gradB,
                         const DeviceBuffer<scalar>* const* UbStored)
{
    const int nC = dm.nCells, nB = dm.nBndFaces;
    DeviceBuffer<scalar> uxb, uyb, uzb;
    DeviceBuffer<scalar>* ub[3] = {&uxb, &uyb, &uzb};
    const DeviceBuffer<scalar>* Uc[3] = {&Ux, &Uy, &Uz};
    // The same choice deviceDivDevReff makes for its own gradB: the value the patch HOLDS when the
    // caller keeps one, since that is what OF's fvc::grad(U) reads (see the header).
    for (int i = 0; i < 3; ++i)
    {
        if (UbStored && UbStored[i] && UbStored[i]->size() == static_cast<std::size_t>(nB))
            deviceCopy(*ub[i], *UbStored[i]);
        else
            deviceBCValue(dbU.comp[i], *Uc[i], *ub[i]);
    }
    gradB.resize(static_cast<std::size_t>(9) * nB);
    {
        const label *bndCelld = dm.bndCell.data(), *bndGFaced = dm.bndGFace.data();
        const scalar *Sfxd = dm.Sfx.data(), *Sfyd = dm.Sfy.data(), *Sfzd = dm.Sfz.data();
        const scalar *gradUd = gradU.data(), *uxbd = uxb.data(), *uybd = uyb.data(), *uzbd = uzb.data();
        const scalar *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data(), *dcd = dbU.comp[0].deltaCoeffs.data();
        const BndSnGradView bx = snGradView(dbU.comp[0]), by = snGradView(dbU.comp[1]), bz = snGradView(dbU.comp[2]);
        scalar* gradBd = gradB.data();
        pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () {
            gradBKernel(nB, bndCelld, bndGFaced, Sfxd, Sfyd, Sfzd, gradUd, nC, uxbd, uybd, uzbd, Uxd, Uyd, Uzd, dcd,
                        bx, by, bz, gradBd);
        });
    }
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
    {
        const label *ownerStart = dm.ownerStart.data(), *losort = dm.losort.data(), *losortStart = dm.losortStart.data();
        const label *own = dm.owner.data(), *nei = dm.nei.data();
        const scalar *wd = dm.w.data(), *Sfxd = dm.Sfx.data(), *Sfyd = dm.Sfy.data(), *Sfzd = dm.Sfz.data();
        const label *bndCellStart = dm.bndCellStart.data(), *bndPerm = dm.bndPerm.data();
        const label *bndGFace = dm.bndGFace.data(), *bndIsEmpty = dm.bndIsEmpty.data();
        const scalar *Tcelld = Tcell.data(), *Tbndd = Tbnd.data(), *Vd = dm.V.data();
        scalar *srcXd = srcX.data(), *srcYd = srcY.data(), *srcZd = srcZ.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            tensorDivKernel(nC, ownerStart, losort, losortStart, own, nei, wd, Sfxd, Sfyd, Sfzd,
                            bndCellStart, bndPerm, bndGFace, bndIsEmpty, Tcelld, Tbndd, nB, Vd, srcXd, srcYd, srcZd);
        });
    }
    cudaCheck(cudaGetLastError(), "tensorDivSource");
    if (cyc) interfaceAddTensorDiv(*cyc, Tcell, nC, srcX, srcY, srcZ);
    if (ami) interfaceAddTensorDiv(*ami, Tcell, nC, srcX, srcY, srcZ);
}


namespace {
// FP-10: the stress term's temporaries persist too, for the reason the momentum assembly's do
// (rhoUEqn.cu). This function is called from inside that assembly, so its buffers have to be stable
// before the phase can be captured: capturing with pool temporaries dies on the first replay with an
// illegal memory access, which is how this was found.
struct DevReffWorkspace
{
    DeviceBuffer<scalar> gradU;
    DeviceBuffer<scalar> sigmaC;
    DeviceBuffer<scalar> gradB;
    DeviceBuffer<scalar> sigmaB;
    DeviceBuffer<scalar> amiUNx;
    DeviceBuffer<scalar> amiUNy;
    DeviceBuffer<scalar> amiUNz;
    DeviceBuffer<scalar> bvals[3];    // also the boundary values the snGrad half reads
    DeviceBuffer<scalar> gxs[3];
    DeviceBuffer<scalar> gys[3];
    DeviceBuffer<scalar> gzs[3];
};
DevReffWorkspace& devReffWorkspace(const DeviceMesh& dm)
{
    static auto& cache = *new std::map<const void*, DevReffWorkspace>();
    return cache[dm.owner.data()];
}
}   // namespace

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
    scalar gradULimitK,
    bool gradULeastSq)
{
    const int nC = dm.nCells, nB = dm.nBndFaces;
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };
    if (gradULeastSq && (cyc || ami || proc))
        throw std::runtime_error("deviceDivDevReff: grad(U) leastSquares is not computed across a coupled interface.");

    // gradU (packed 9*nC): row i = gaussGrad(U_i).
    DevReffWorkspace& ws = devReffWorkspace(dm);       // FP-10: persistent temporaries, see above
    DeviceBuffer<scalar>& gradU = ws.gradU;
    gradU.resize(static_cast<std::size_t>(9) * nC);
    // The boundary values ARE the per-component buffers below: this used to hold three more and take
    // ownership of them with std::move, which swapped their device pointers every call -- the one
    // thing a captured graph cannot survive. Aliasing them costs nothing and reads the same values.
    DeviceBuffer<scalar>& uxb = ws.bvals[0];
    DeviceBuffer<scalar>& uyb = ws.bvals[1];
    DeviceBuffer<scalar>& uzb = ws.bvals[2];

    // rotational AMI: the gradU neighbour is the ROTATED AMI-interpolated U (forwardT.interp), precompute it.
    DeviceBuffer<scalar>& amiUNx = ws.amiUNx;
    DeviceBuffer<scalar>& amiUNy = ws.amiUNy;
    DeviceBuffer<scalar>& amiUNz = ws.amiUNz;
    if (ami && ami->rotational)
        deviceAmiInterpolateVec(*ami, Ux, Uy, Uz, amiUNx, amiUNy, amiUNz);
    DeviceBuffer<scalar>* amiUN[3] = { &amiUNx, &amiUNy, &amiUNz };

    // gaussGrad(U_i) = (dUi/dx, dUi/dy, dUi/dz) = COLUMN i of G (OF convention G_ij = dU_j/dx_i, via outer(Sf,U)).
    //
    // The three boundary values are built FIRST, in the same per-component order and with the same halo
    // exchange (whose receive buffer one component reuses from the next, so the exchanges must stay
    // sequential), and the three gradients then come from ONE fused launch. The gradient kernel re-reads
    // the whole mesh addressing and geometry per launch while the field it differentiates is a small
    // part of that traffic, so three launches moved it three times: measured at 305,760 cells, three
    // separate calls cost 864 us against the fused one's 389. Bit-identical per component
    // (tests/test_grad_fused.cu holds the fused kernel to three separate deviceGaussGrad calls by
    // memcmp). The per-component work AFTER the gradient -- the cyclic and AMI contributions, which
    // OpenFOAM adds to the base Gauss gradient before any limiting -- is unchanged and still per i.
    DeviceBuffer<scalar>* bvals = ws.bvals;
    for (int i = 0; i < 3; ++i)
    {
        // U's boundary as OF's fvc::grad(U) reads it: the STORED value when the caller keeps one, and a
        // re-derivation only when it does not. The two agree only while the caller evaluates U's boundary
        // before every assembly, which OpenFOAM does not -- see the header (queue items 25, 30).
        if (UbStored && UbStored[i] && UbStored[i]->size() == static_cast<std::size_t>(nB))
            deviceCopy(bvals[i], *UbStored[i]);
        else
            deviceBCValue(dbU.comp[i], *Uc[i], bvals[i]);
        if (proc)   // processor faces are bcType 8: deviceBCValue leaves them, the halo supplies the value
        {
            proc->halo->exchange(Uc[i]->data());
            proc->halo->scatterBoundaryValues(Uc[i]->data(), *proc->weights, *proc->procStart, bvals[i].data());
            proc->halo->waitExchange();   // recv buffer is reused by the next component -- see device_halo.cuh
        }
    }
    DeviceBuffer<scalar>* gxs = ws.gxs;
    DeviceBuffer<scalar>* gys = ws.gys;
    DeviceBuffer<scalar>* gzs = ws.gzs;
    // fvc::grad(U) through the case's grad(U) entry (linearViscousStress.C's divDevRhoReff takes
    // dev2(T(fvc::grad(U)))): leastSquares where it resolves so, the host's gradULeastSq.
    {
        const DeviceBuffer<scalar>* vol[3] = {Uc[0], Uc[1], Uc[2]};
        const DeviceBuffer<scalar>* bv[3]  = {&bvals[0], &bvals[1], &bvals[2]};
        if (gradULeastSq) deviceLeastSquaresGradFused(dm, 3, vol, bv, gxs, gys, gzs);   // FP-3: one launch
        else              deviceGaussGradFused(dm, 3, vol, bv, gxs, gys, gzs);
    }
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar>& bval = bvals[i];
        DeviceBuffer<scalar>& gx = gxs[i];
        DeviceBuffer<scalar>& gy = gys[i];
        DeviceBuffer<scalar>& gz = gzs[i];

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

        // ASYNC on the per-thread stream, ordered before the consumer, as deviceGradU and
        // deviceLeastSquaresGradU do it. These were BLOCKING copies -- nine per momentum assembly, each
        // draining the queue and leaving the host to refill it, inside a phase measured at 59% GPU-busy
        // -- and a blocking copy is also what makes this function impossible to capture into a graph,
        // which is where FP-10's row is headed. Same bytes, same order, same bits.
        cudaCheck(cudaMemcpyAsync(gradU.data() + (0*3+i)*nC, gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "ddr g");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (1*3+i)*nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "ddr g");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (2*3+i)*nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "ddr g");
    }

    // ...then LIMIT it, if the case named a limited grad(U). OF applies cellLimitedGrad to the base
    // Gauss gradient AFTER the coupled-patch contributions are in, which is exactly here.
    if (gradULimitK > scalar(0)) deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradU, gradULimitK, cyc, ami);

    // sigma cell
    DeviceBuffer<scalar>& sigmaC = ws.sigmaC;
    sigmaC.resize(static_cast<std::size_t>(9) * nC);
    {
        const scalar *gradUd = gradU.data(), *nuCelld = nuCell.data(); scalar* sigmaCd = sigmaC.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { sigmaKernel(nC, gradUd, nuCelld, sigmaCd); });
    }
    cudaCheck(cudaGetLastError(), "ddr sigmaC");

    // boundary gradient + sigma boundary
    DeviceBuffer<scalar>& gradB = ws.gradB;
    gradB.resize(static_cast<std::size_t>(9) * nB);
    {
        const label *bndCelld = dm.bndCell.data(), *bndGFaced = dm.bndGFace.data();
        const scalar *Sfxd = dm.Sfx.data(), *Sfyd = dm.Sfy.data(), *Sfzd = dm.Sfz.data();
        const scalar *gradUd = gradU.data(), *uxbd = uxb.data(), *uybd = uyb.data(), *uzbd = uzb.data();
        const scalar *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data(), *dcd = dbU.comp[0].deltaCoeffs.data();
        const BndSnGradView bx = snGradView(dbU.comp[0]), by = snGradView(dbU.comp[1]), bz = snGradView(dbU.comp[2]);
        scalar* gradBd = gradB.data();
        pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () {
            gradBKernel(nB, bndCelld, bndGFaced, Sfxd, Sfyd, Sfzd, gradUd, nC, uxbd, uybd, uzbd, Uxd, Uyd, Uzd, dcd,
                        bx, by, bz, gradBd);
        });
    }
    cudaCheck(cudaGetLastError(), "ddr gradB");
    DeviceBuffer<scalar>& sigmaB = ws.sigmaB;
    sigmaB.resize(static_cast<std::size_t>(9) * nB);
    {
        const scalar *gradBd = gradB.data(), *nuBndd = nuBnd.data(); scalar* sigmaBd = sigmaB.data();
        pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () { sigmaKernel(nB, gradBd, nuBndd, sigmaBd); });
    }
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
    {
        const label *ownerStart = dm.ownerStart.data(), *losort = dm.losort.data(), *losortStart = dm.losortStart.data();
        const label *own = dm.owner.data(), *nei = dm.nei.data();
        const scalar *wd = dm.w.data(), *Sfxd = dm.Sfx.data(), *Sfyd = dm.Sfy.data(), *Sfzd = dm.Sfz.data();
        const label *bndCellStart = dm.bndCellStart.data(), *bndPerm = dm.bndPerm.data();
        const label *bndGFace = dm.bndGFace.data(), *bndIsEmpty = dm.bndIsEmpty.data();
        const scalar *sigmaCd = sigmaC.data(), *sigmaBd = sigmaB.data(), *Vd = dm.V.data();
        scalar *srcXd = srcX.data(), *srcYd = srcY.data(), *srcZd = srcZ.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            tensorDivKernel(nC, ownerStart, losort, losortStart, own, nei, wd, Sfxd, Sfyd, Sfzd,
                            bndCellStart, bndPerm, bndGFace, bndIsEmpty, sigmaCd, sigmaBd, nB, Vd, srcXd, srcYd, srcZd);
        });
    }
    cudaCheck(cudaGetLastError(), "ddr tensorDiv");
    if (cyc) interfaceAddTensorDiv(*cyc, sigmaC, nC, srcX, srcY, srcZ);   // + cyclic-face stress flux (V*fvc::div)
    if (ami) interfaceAddTensorDiv(*ami, sigmaC, nC, srcX, srcY, srcZ);      // + AMI-face stress flux
}

} // namespace brae
