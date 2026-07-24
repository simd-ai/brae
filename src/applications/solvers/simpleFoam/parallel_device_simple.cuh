#pragma once
// brae distributed device SIMPLE, the multi-GPU laminar SIMPLE step composed from the distributed device
// primitives (deviceParallelAmul, deviceParallelJacobiPCG, DeviceHalo). Built increment by increment; kept
// SEPARATE from the single-GPU DeviceSimpleSolver so it cannot regress the OpenFOAM-validated solver.
//
// Processor faces are a coupled interface (the DeviceCyclic/DeviceAMI pattern): the matrix path adds an
// off-diagonal interface coeff (this file), and the explicit operators inject the coupled face value into the
// boundary array (DeviceHalo::scatterBoundaryValues). A processor face is NEVER a real boundary face.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_boundary.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"          // local per-rank AMG preconditioner for the distributed pressure solve
#include "device_divdevreff.cuh"
#include "device_halo.cuh"
#include "parallel_simple.cuh"     // Partition, distributeFromCells
#include "reconstruct.cuh"
#include "local_assembly.cuh"   // computeProcUpwindD
#include "device_kepsilon.cuh"   // DeviceWallData / KEpsilonCoeffs / buildDeviceWallData (Phase 4b turbulence)
#include "komega_sst_coeffs.cuh"
#include "device_komega_sst.cuh" // deviceS2/deviceF2/deviceNutSST for validateTurbulence()'s SST nut seeding
#include "cell_wall_dist.cuh"
#include "parallel_device_interface.cuh"   // deviceMomentumInterface / deviceInterfaceOffDiagSum (shared)
#include "device_mrf.cuh"          // DeviceMRF: Coriolis source + makeRelative frame flux (rotating cell zone)
#include "parallel_device_ami.cuh" // DistributedAMI + buildDistributedAMI (cyclicAMI on the distributed path, WIP)
#include "fv_options.cuh"          // FvOptionsData (per-cell momentum Su/Sp + porosity)
#include "device_fvoptions.cuh"    // DevicePorosity + deviceFvoPorosityDiag/Source
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <filesystem>

namespace brae {

namespace detail {

// Stage-1 OVERLAP: fill the extended matrix's interface off-diagonal block, dst[i] = -src[i]
// (upper/lower on the owned<->ghost face = a_og = -ifCoeff, matching the global matvec's -ifCoeff*psiNbr).
__global__
void negCopyK(scalar* __restrict__ dst, const scalar* __restrict__ src, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = -src[i];
}

// L4: the processor contribution to H(). lduMatrix::H(psi) = -offdiag.psi, and the PARALLEL A.psi carries
// -ifCoeff*psiNbr at each interface face, so H picks up +ifCoeff*psiNbr there. deviceMatrixH already divided
// by V, hence the /V. atomicAdd: a cell may own several faces on one interface.
__global__
void matrixHInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ psiNbr,
    const scalar* __restrict__ V,
    scalar*       __restrict__ Hk,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const label c = faceCells[f];
    atomicAdd(&Hk[c], ifCoeff[f] * psiNbr[f] / V[c]);
}

// L7: the pressure flux across a processor face, pEqn.flux() -- mirrors host parallelMatrixFlux:
//   flux[f] = -ifCoeff[f] * (p_neighbour[f] - p_local[faceCells[f]])
// It is conservative because the laplacian interface coeff has the same magnitude on both sides, so the two
// sides differ only by the swap of (p_nbr, p_local) -> equal and opposite. fluxOut points at the interface's
// slice of the flattened boundary-flux array.
__global__
void matrixFluxInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ pNbr,
    const scalar* __restrict__ p,
    scalar*       __restrict__ fluxOut,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    fluxOut[f] = -ifCoeff[f] * (pNbr[f] - p[faceCells[f]]);
}

// The processor-interface coeffs of a laplacian(gamma, .) matrix -- the pressure equation's coupling. Mirrors
// host assembleLocalLaplacianF, per cut face f (coeff = gammaF*magSf*procDelta):
//   diag[faceCells[f]] -= coeff[f]      (atomic: a cell may own several interface faces)
//   ifCoeff[f]          = -coeff[f]
__global__
void laplacianInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ coeff,
    scalar*       __restrict__ diag,
    scalar*       __restrict__ ifCoeff,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    ifCoeff[f] = -coeff[f];
    atomicAdd(&diag[faceCells[f]], -coeff[f]);
}

// linearUpwind deferred correction at a PROCESSOR face -- the distributed analogue of
// linearUpwindCorrKernel (device_simple.cu) and of deviceCyclicAddLinUpwindCorr (no rotation here: a
// processor cut is a plain interior face, the two sides share an orientation).
//
// The matrix stays plain UPWIND; linearUpwind is this explicit source. Per cut face, the local cell is the
// face's owner and phi is the OUTWARD flux, so it is the owner branch of the internal kernel:
//   corr[localCell] += phi * (grad(U_comp)[upwind] . d_upwind)
// with the upwind side chosen by sign(phi):
//   phi >= 0 -> LOCAL is upwind:  grad at the local cell,  d = dOwn = Cf - C[local]
//   phi <  0 -> REMOTE is upwind: grad at the remote cell (gxN/gyN/gzN, from the halo), d = dNei = Cf - C[remote]
// dOwn/dNei are static geometry, precomputed once by host computeProcUpwindD (the remote centre needs an
// exchange). The caller does relaxSrc -= corr, exactly as for the internal correction.
__global__
void linUpwindInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ gxN,
    const scalar* __restrict__ gyN,
    const scalar* __restrict__ gzN,
    const scalar* __restrict__ dOwn,
    const scalar* __restrict__ dNei,
    scalar*       __restrict__ corr,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const scalar pf = phi[f];
    const label  c  = faceCells[f];
    scalar g[3], d[3];
    if (pf >= 0.0)   // local upwind: local gradient, local offset
    {
        g[0] = gx[c];  g[1] = gy[c];  g[2] = gz[c];
        d[0] = dOwn[3*f]; d[1] = dOwn[3*f+1]; d[2] = dOwn[3*f+2];
    }
    else             // remote upwind: the halo gradient at the remote cell, remote offset
    {
        g[0] = gxN[f]; g[1] = gyN[f]; g[2] = gzN[f];
        d[0] = dNei[3*f]; d[1] = dNei[3*f+1]; d[2] = dNei[3*f+2];
    }
    atomicAdd(&corr[c], pf * (g[0]*d[0] + g[1]*d[1] + g[2]*d[2]));   // a cell may own several cut faces
}

// LUST linear part at a PROCESSOR face -- the distributed analogue of linearCorrKernel (device_simple.cu).
// The face correction is (linear - upwind) weighted by phi; with linear = w*own + (1-w)*nbr and
// upwind = pos0*own + (1-pos0)*nbr (pos0 = phi>=0), that difference collapses to (w - pos0)*(own - nbr):
//   corr[localCell] += phi * (w - pos0) * (psi[local] - psi[remote])
// The local cell is the cut face's owner and phi is the OUTWARD flux, so this is the internal kernel's owner
// branch. `w` is procW -- the LOCAL side's interpolation weight, matching linearCorrKernel's w[f] = w of own.
// NB the single-GPU path OMITS the cyclic linear part ("LUST cases are non-cyclic"); that shortcut is NOT
// available here, since a decomposed mesh has processor faces everywhere.
__global__
void linearCorrInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ w,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ psiNbr,
    scalar*       __restrict__ corr,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const scalar pf   = phi[f];
    const scalar pos0 = (pf >= 0.0) ? 1.0 : 0.0;
    const label  c    = faceCells[f];
    atomicAdd(&corr[c], pf * (w[f] - pos0) * (psi[c] - psiNbr[f]));   // a cell may own several cut faces
}

// Non-orthogonal ("corrected") laplacian correction at a PROCESSOR face -- the distributed analogue of
// lapCorrFaceKernel + lapCorrGatherKernel (device_fvm.cu) and of deviceCyclicAddLapCorr (no rotation here).
// Mirrors the internal formula exactly, with the local cell as owner and the remote as neighbour:
//     grad_face = w*grad[local] + (1-w)*grad[remote]          (w = procW, the LOCAL/owner weight)
//     ffc       = gamma_f * magSf * (corrVec . grad_face)
//     corr[local] -= ffc                                      (the gather kernel's OWNER branch: s -= ffc)
// gamma_f is the face value of the diffusivity (nuEff for momentum, rAU for the pEqn) -- already the
// halo-interpolated boundary value. corrVec/magSf are static geometry from host computeProcNonOrth.
// On an ORTHOGONAL mesh corrVec == 0, so this contributes exactly nothing -- which is why it is invisible
// there, and why it must be validated on the SHEARED duct (see test_gpu_parallel_duct).
__global__
void lapCorrInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ gammaF,
    const scalar* __restrict__ magSfF,
    const scalar* __restrict__ corrVec,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ gxN,
    const scalar* __restrict__ gyN,
    const scalar* __restrict__ gzN,
    scalar*       __restrict__ corr,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const label  c  = faceCells[f];
    const scalar wf = w[f], wn = 1.0 - wf;
    const scalar gfx = wf * gx[c] + wn * gxN[f];
    const scalar gfy = wf * gy[c] + wn * gyN[f];
    const scalar gfz = wf * gz[c] + wn * gzN[f];
    const scalar ffc = gammaF[f] * magSfF[f]
                     * (corrVec[3*f] * gfx + corrVec[3*f+1] * gfy + corrVec[3*f+2] * gfz);
    atomicAdd(&corr[c], -ffc);   // owner branch; a cell may own several cut faces
}

// PRESSURE non-orthogonal correction at a PROCESSOR face -- the distributed analogue of deviceCyclicLapCorrP.
// Does BOTH jobs, exactly as the internal path does with deviceLaplacianCorrFlux + deviceFaceDivSource:
//   ffc         = rAU_f * magSf * (corrVec . grad(p)_face)      (grad_face = w*grad[local] + (1-w)*grad[remote])
//   pb[local]  -= ffc      (the faceDivSource OWNER branch: b += -V*div(ffc))
//   ffcOut[f]   = ffc      (the caller then does phi -= ffc, or continuity breaks on non-orthogonal cut faces)
// Both are required: the source alone leaves the reconstructed flux non-conservative.
__global__
void lapCorrPInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ gammaF,
    const scalar* __restrict__ magSfF,
    const scalar* __restrict__ corrVec,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ gxN,
    const scalar* __restrict__ gyN,
    const scalar* __restrict__ gzN,
    scalar*       __restrict__ pb,
    scalar*       __restrict__ ffcOut,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const label  c  = faceCells[f];
    const scalar wf = w[f], wn = 1.0 - wf;
    const scalar gfx = wf * gx[c] + wn * gxN[f];
    const scalar gfy = wf * gy[c] + wn * gyN[f];
    const scalar gfz = wf * gz[c] + wn * gzN[f];
    const scalar ffc = gammaF[f] * magSfF[f]
                     * (corrVec[3*f] * gfx + corrVec[3*f+1] * gfy + corrVec[3*f+2] * gfz);
    ffcOut[f] = ffc;
    atomicAdd(&pb[c], -ffc);
}

// dst[offset + f] -= src[f]  (subtract an interface's flux correction out of the flattened boundary-flux array)
__global__
void subtractSliceKernel(scalar* __restrict__ dst, int offset, const scalar* __restrict__ src, int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    dst[offset + f] -= src[f];
}

// out[f] = bndField[offset + f] * geom[f]  (the processor-interface coupling coefficient, on-device). Replaces the
// per-interface host loop `coeff = fieldAtCut * magSf * procDelta` that forced a full .host() D2D sync of the cut
// field every SIMPLE step: fieldAtCut is already a DEVICE boundary array (halo-scattered) and geom = magSf*delta is
// static (built once in the ctor), so the whole coefficient is a fused device multiply -- no round-trip.
__global__
void interfaceCoeffKernel(
    scalar*       __restrict__ out,
    const scalar* __restrict__ bndField,
    int offset,
    const scalar* __restrict__ geom,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    out[f] = bndField[offset + f] * geom[f];
}

} // namespace detail

// Add the processor faces' linearUpwind deferred correction into `corr` (which already holds the internal
// deviceLinearUpwindCorr result). `gx/gy/gz` are grad(U_comp) as CELL fields -- already processor-consistent
// (the halo injected the coupled face value before the gaussGrad). The three exchanges below fetch that same
// gradient at the REMOTE cell, needed only where the remote side is upwind.
// `phiF[i]` is the processor-face flux, `dOwnD/dNeiD[i]` the precomputed offsets (3 per face).
inline void deviceLinUpwindInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    const std::vector<DeviceBuffer<scalar>>& dOwnD,
    const std::vector<DeviceBuffer<scalar>>& dNeiD,
    DeviceBuffer<scalar>& corr,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    // The recv buffer is shared and reused, so each component's remote gradient must be COPIED out before the
    // next exchange overwrites it (see the hazard note in device_halo.cuh).
    const int nI = halo.nInterfaces();
    std::vector<DeviceBuffer<scalar>> gN[3];
    const DeviceBuffer<scalar>* gsrc[3] = { &gx, &gy, &gz };
    for (int k = 0; k < 3; ++k)
    {
        halo.exchange(gsrc[k]->data(), stream);
        gN[k].resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            const int n = static_cast<int>(halo.size(i));
            if (n <= 0) continue;
            gN[k][i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(gN[k][i].data(), halo.recvData(i), n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, stream),
                "linUpwind halo grad copy");
        }
        halo.waitExchange(stream);
    }
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::linUpwindInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            phiF[i].data(),
            gx.data(),
            gy.data(),
            gz.data(),
            gN[0][i].data(),
            gN[1][i].data(),
            gN[2][i].data(),
            dOwnD[i].data(),
            dNeiD[i].data(),
            corr.data(),
            n);
    }
}

// Add the processor faces' LUST linear-part correction into `corr` (which already holds the internal
// deviceLinearCorr result). Only the remote psi is needed, so this is a single exchange per component.
inline void deviceLinearCorrInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>& corr,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    halo.exchange(psi.data(), stream);
    for (int i = 0; i < halo.nInterfaces(); ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::linearCorrInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            phiF[i].data(),
            weights[i].data(),
            psi.data(),
            halo.recvData(i),
            corr.data(),
            n);
    }
    halo.waitExchange(stream);   // the kernels above READ the shared recv buffer -- see device_halo.cuh
}

// Add the processor faces' non-orthogonal laplacian correction into `corr` (which already holds the internal
// deviceLaplacianCorr result). Needs grad(psi) at the REMOTE cell, so it exchanges the 3 gradient components --
// the same pattern as deviceLinUpwindInterface, copying each out of the shared recv buffer before the next
// exchange overwrites it (see the hazard note in device_halo.cuh).
inline void deviceLapCorrInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& gammaF,
    const std::vector<DeviceBuffer<scalar>>& magSfF,
    const std::vector<DeviceBuffer<scalar>>& corrVec,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& corr,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    const int nI = halo.nInterfaces();
    std::vector<DeviceBuffer<scalar>> gN[3];
    const DeviceBuffer<scalar>* gsrc[3] = { &gx, &gy, &gz };
    for (int k = 0; k < 3; ++k)
    {
        halo.exchange(gsrc[k]->data(), stream);
        gN[k].resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            const int n = static_cast<int>(halo.size(i));
            if (n <= 0) continue;
            gN[k][i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(gN[k][i].data(), halo.recvData(i), n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, stream),
                "lapCorr halo grad copy");
        }
        halo.waitExchange(stream);
    }
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::lapCorrInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            gammaF[i].data(),
            magSfF[i].data(),
            corrVec[i].data(),
            weights[i].data(),
            gx.data(), gy.data(), gz.data(),
            gN[0][i].data(), gN[1][i].data(), gN[2][i].data(),
            corr.data(),
            n);
    }
}

// The processor faces' PRESSURE non-orth correction: folds -ffc into `pb` and returns ffc per interface so the
// caller can keep the reconstructed flux conservative. `gammaF[i]` is rAU at interface i's faces.
inline void deviceLapCorrPInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& gammaF,
    const std::vector<DeviceBuffer<scalar>>& magSfF,
    const std::vector<DeviceBuffer<scalar>>& corrVec,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& pb,
    std::vector<DeviceBuffer<scalar>>& ffcOut,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    const int nI = halo.nInterfaces();
    std::vector<DeviceBuffer<scalar>> gN[3];
    const DeviceBuffer<scalar>* gsrc[3] = { &gx, &gy, &gz };
    for (int k = 0; k < 3; ++k)
    {
        halo.exchange(gsrc[k]->data(), stream);
        gN[k].resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            const int n = static_cast<int>(halo.size(i));
            if (n <= 0) continue;
            gN[k][i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(gN[k][i].data(), halo.recvData(i), n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, stream),
                "lapCorrP halo grad copy");
        }
        halo.waitExchange(stream);
    }
    ffcOut.resize(nI);
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        ffcOut[i].resize(n);
        detail::lapCorrPInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            gammaF[i].data(),
            magSfF[i].data(),
            corrVec[i].data(),
            weights[i].data(),
            gx.data(), gy.data(), gz.data(),
            gN[0][i].data(), gN[1][i].data(), gN[2][i].data(),
            pb.data(),
            ffcOut[i].data(),
            n);
    }
}


// Assemble a laplacian(gamma, .) matrix's processor coupling (the pressure equation): fold -coeff into `diag`
// and produce `ifCoeff[i]` per interface. `coeffGeo[i] = gammaF*magSf*procDelta` of interface i.
inline void deviceLaplacianInterface(
    const DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& coeffGeo,
    DeviceBuffer<scalar>& diag,
    std::vector<DeviceBuffer<scalar>>& ifCoeff,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    const int nI = halo.nInterfaces();
    ifCoeff.resize(nI);
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        ifCoeff[i].resize(n);
        detail::laplacianInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            coeffGeo[i].data(),
            diag.data(),
            ifCoeff[i].data(),
            n);
    }
}

// Write the processor-face pressure flux into `fluxB` (the flattened boundary-flux array) at each interface's
// procStart offset. Exchanges p itself. The result cancels across a cut face (both sides equal and opposite),
// which is what keeps the corrected phi globally conservative.
inline void deviceParallelMatrixFluxInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& ifCoeff,
    const std::vector<label>& procStart,
    const DeviceBuffer<scalar>& p,
    DeviceBuffer<scalar>& fluxB,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    halo.exchange(p.data(), stream);
    for (int i = 0; i < halo.nInterfaces(); ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::matrixFluxInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            ifCoeff[i].data(),
            halo.recvData(i),
            p.data(),
            fluxB.data() + procStart[i],
            n);
    }
    halo.waitExchange(stream);   // protect the recv buffer from the next exchange (see device_halo.cuh hazard)
}

// Max mesh non-orthogonality in DEGREES: the angle between a face's Sf and its owner->neighbour centre vector,
// over internal AND processor faces (the remote centre comes from the same exchange computeProcUpwindD uses).
// OF's "corrected"/"limited" laplacian adds an explicit correction that scales with this angle; the distributed
// device path does NOT implement that correction at processor faces, so on a non-orthogonal mesh it would
// silently solve the "orthogonal" scheme instead -- converged, plausible and WRONG. On an orthogonal mesh the
// correction is identically zero, so the scheme is safe to accept there. This measures which case we are in.
inline scalar maxNonOrthogonality(const Partition& P)
{
    const PrimitiveMesh& m = P.Lm.mesh;
    const FvGeometry& g = P.lg;
    const scalar rad2deg = 180.0 / 3.14159265358979323846;
    auto angle = [&](const vector& S, const vector& d) -> scalar
    {
        const scalar den = mag(S) * mag(d);
        if (den <= 0) return 0.0;
        const scalar c = std::fmin(1.0, std::fmax(-1.0, dot(S, d) / den));
        return std::acos(c) * rad2deg;
    };
    scalar mx = 0;
    for (label f = 0; f < m.nInternalFaces(); ++f)
        mx = std::fmax(mx, angle(g.Sf()[f], g.C()[m.neighbour()[f]] - g.C()[m.owner()[f]]));

    std::vector<std::vector<scalar>> dO, dN;   // dOwn = Cf-C_local, dNei = Cf-C_remote
    computeProcUpwindD(P.Lm, P.lg, P.lp, dO, dN);
    std::size_t j = 0;
    for (std::size_t pi = 0; pi < P.lp.size(); ++pi)
    {
        if (P.lp[pi].type != "processor") continue;
        for (label i = 0; i < P.lp[pi].size; ++i)
        {
            const vector d{ dO[j][3*i]   - dN[j][3*i],        // = C_remote - C_local
                            dO[j][3*i+1] - dN[j][3*i+1],
                            dO[j][3*i+2] - dN[j][3*i+2] };
            mx = std::fmax(mx, angle(g.Sf()[P.lp[pi].start + i], d));
        }
        ++j;
    }
    return Pstream::allReduce(mx, ReduceOp::Max);
}

// ----------------------------------------------------------------------------------------------------------
// ParallelDeviceSimple: the closed distributed laminar SIMPLE loop. One rank == one partition == one GPU.
//
// U/p/phi stay on the device between iterations; step() runs one SIMPLE iteration entirely on the GPU:
//   assemble M = div(phi,U) - laplacian(nuEff,U) (+ processor interface)  ->  relax  ->  momentum predictor
//   rAU / H() / HbyA  ->  phiHbyA  ->  pEqn laplacian(rAU,p) == div(phiHbyA)  ->  conservative phi + corrector
// Each stage is the one validated against host parallelSimpleStepLaminar in test_gpu_parallel_predictor.
//
// Laminar only (nuEff = nu); turbulence is Phase 4b. Processor faces are COUPLED: zero matrix coeffs (bcType 8)
// with the halo-interpolated face value injected for the explicit operators.
// ----------------------------------------------------------------------------------------------------------
class ParallelDeviceSimple
{
public:
    ParallelDeviceSimple(
        const Partition& part,
        const GeometricField<vector>& U0,
        const GeometricField<scalar>& p0,
        scalar nu,
        scalar relaxU,
        scalar relaxP,
        scalar tolU,
        scalar tolP,
        int maxIter,
        bool bounded = false,
        bool linearUpwind = false,
        bool lust = false,
        bool linUpwindV = false,               // div(phi,U) "linearUpwindV": linearUpwind + OF's vector limiter
        bool nonOrth = false,
        bool consistent = false,               // SIMPLEC (fvSolution SIMPLE.consistent): rAtU consistency correction
        bool limitedU = false,                 // div(phi,U) "limitedLinear" (magSqr): implicit TVD limited convection
        scalar limitedTwoByk = 2.0,            // 2/max(k_,SMALL) from the scheme coefficient (limitedLinear 1 -> 2)
        bool limitedV = false,                 // div(phi,U) "limitedLinearV" (NVDVTVDV vector limiter): implicit
        const std::string& amgCacheDir = "",   // polyMesh dir for the per-rank AMG-hierarchy disk cache ("" = off)
        // INEXACT SIMPLE. OF -- and brae's own single-GPU path (gpuSimpleFoam.cu reads solvers/<f>/relTol) --
        // solve each equation only to a RELATIVE tolerance per outer iteration, because an outer iteration is
        // itself an approximation: driving momentum and pressure to full ABSOLUTE tolerance every step is
        // wasted work. This path used to hardcode 0.0 at the three solve call sites, so it over-solved both
        // (measured: ~340-560 combined Krylov iterations per SIMPLE step on a 391k-cell/rank pitzDaily).
        // Default stays 0.0 so the np=1-vs-np>1 correctness tests keep their tight, fully-converged compare.
        scalar relTolU = 0.0,
        scalar relTolP = 0.0,
        scalar relTolKE = 0.0)   // k/epsilon/omega/nuTilda transport (OF: min of the turbulence fields' relTol)
        : P_(part),
          nu_(nu),
          relaxU_(relaxU),
          relaxP_(relaxP),
          tolU_(tolU),
          tolP_(tolP),
          relTolU_(relTolU),
          relTolP_(relTolP),
          relTolKE_(relTolKE),
          maxIter_(maxIter),
          bounded_(bounded),
          linearUpwind_(linearUpwind),
          lust_(lust),
          linUpwindV_(linUpwindV),
          nonOrth_(nonOrth),
          consistent_(consistent),
          limitedU_(limitedU),
          twoByk_(limitedTwoByk),
          limitedV_(limitedV),
          lnC_(part.Lm.mesh.nCells()),
          nIf_(part.Lm.mesh.nInternalFaces()),
          dm_(buildDeviceMesh(part.Lm.mesh, part.lg, part.lp)),
          dbU_(buildDeviceVectorBoundary(U0, part.lp, part.lg)),
          dbP_(buildDeviceBoundary(p0, part.lp, part.lg)),
          halo_(part.rank, intNbrs(part), part.Lm.procFaceCells)
    {
        const std::vector<FvPatch>& lp = P_.lp;
        // initial device state
        std::vector<scalar> ux(lnC_), uy(lnC_), uz(lnC_);
        for (label c = 0; c < lnC_; ++c)
        {
            ux[c] = U0.internal[c].x;
            uy[c] = U0.internal[c].y;
            uz[c] = U0.internal[c].z;
        }
        Uk_[0].copyFrom(ux);
        Uk_[1].copyFrom(uy);
        Uk_[2].copyFrom(uz);
        dp_.copyFrom(p0.internal);
        ones_.copyFrom(std::vector<scalar>(lnC_, 1.0));
        zeroSrc_.copyFrom(std::vector<scalar>(lnC_, 0.0));

        // validComponents (fvMesh::validComponents): an empty-patch direction is not solved, so it must not
        // pollute the residualControl measure. Mirrors host parallelSimpleStepLaminar.
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
            if (lp[pi].type == "empty" && lp[pi].size > 0)
            {
                scalar ax = 0, ay = 0, az = 0;
                for (label i = 0; i < lp[pi].size; ++i)
                {
                    const vector& n = P_.lg.Sf()[lp[pi].start + i];
                    ax += std::fabs(n.x);
                    ay += std::fabs(n.y);
                    az += std::fabs(n.z);
                }
                validC_[(ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2)] = false;
            }

        // processor-patch offsets in the flattened boundary array + per-interface device addressing
        label bidx = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            if (lp[pi].type == "processor") procStart_.push_back(bidx);
            bidx += lp[pi].size;
        }
        nBnd_ = dbU_.n;   // == bidx; the flattened boundary array excludes cyclic patches
        zeroBnd_.copyFrom(std::vector<scalar>(nBnd_, 0.0));
        nuEffBnd_.copyFrom(std::vector<scalar>(nBnd_, nu_));
        nuCell_.copyFrom(std::vector<scalar>(lnC_, nu_));
        faceCellsD_.resize(P_.Lm.procFaceCells.size());
        weightsD_.resize(P_.procW.size());
        for (std::size_t i = 0; i < P_.Lm.procFaceCells.size(); ++i) faceCellsD_[i].copyFrom(P_.Lm.procFaceCells[i]);
        for (std::size_t i = 0; i < P_.procW.size(); ++i) weightsD_[i].copyFrom(P_.procW[i]);

        // linearUpwind needs Cf-C on BOTH sides of a cut; the remote centre needs an exchange, but the
        // geometry is static, so do it once here rather than per iteration.
        if (linearUpwind_ || lust_)
        {
            std::vector<std::vector<scalar>> dO, dN;
            computeProcUpwindD(P_.Lm, P_.lg, P_.lp, dO, dN);
            dOwnD_.resize(dO.size());
            dNeiD_.resize(dN.size());
            for (std::size_t i = 0; i < dO.size(); ++i) dOwnD_[i].copyFrom(dO[i]);
            for (std::size_t i = 0; i < dN.size(); ++i) dNeiD_[i].copyFrom(dN[i]);
        }

        // non-orth geometry: static, so build it once. procNonOrth_ REPLACES procDelta in the interface
        // coeff (implicit); corrVecD_ drives the explicit correction. Also cache per-interface magSf and the
        // (constant, laminar) face nuEff, both needed by the correction kernel.
        if (nonOrth_)
        {
            std::vector<std::vector<scalar>> cvH;
            computeProcNonOrth(P_.Lm, P_.lg, P_.lp, procNonOrth_, cvH);
            corrVecD_.resize(cvH.size());
            for (std::size_t i = 0; i < cvH.size(); ++i) corrVecD_[i].copyFrom(cvH[i]);
        }
        {
            std::size_t pj = 0;
            for (std::size_t pi = 0; pi < lp.size(); ++pi)
            {
                if (lp[pi].type != "processor") continue;
                std::vector<scalar> ms(lp[pi].size), nf(lp[pi].size, nu_);
                for (label i = 0; i < lp[pi].size; ++i) ms[i] = P_.lg.magSf()[lp[pi].start + i];
                magSfD_.emplace_back();
                magSfD_.back().copyFrom(ms);
                nuFaceD_.emplace_back();
                nuFaceD_.back().copyFrom(nf);
                // geomIf = magSf * (nonOrth ? procNonOrth : procDelta): the static device geometry factor of the
                // momentum + pEqn interface coefficients (Phase A -- lets those coeffs build on-device, no .host()).
                std::vector<scalar> gi(lp[pi].size);
                for (label i = 0; i < lp[pi].size; ++i)
                    gi[i] = ms[i] * (nonOrth_ ? procNonOrth_[pj][i] : P_.procDelta[pj][i]);
                geomIf_.emplace_back();
                geomIf_.back().copyFrom(gi);
                ++pj;
            }
        }

        // initial conservative flux phi = flux(U0), internal + boundary (processor faces carry the coupled flux)
        const SurfaceScalarField phi0 = fvc::flux(U0, P_.Lm.mesh, P_.lg, P_.lp);
        phiInt_.copyFrom(std::vector<scalar>(phi0.internal.begin(), phi0.internal.begin() + nIf_));
        std::vector<scalar> pb;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            for (label i = 0; i < lp[pi].size; ++i) pb.push_back(phi0.boundary[pi][i]);
        }
        phiBnd_.copyFrom(pb);
        // rAU's boundary shape: zeroGradient on real patches, coupled on processor (mirrors distributeFromCells)
        rAUfld_ = distributeFromCells<scalar>(std::vector<scalar>(lnC_, 1.0), P_);
        dbRAU_  = buildDeviceBoundary(rAUfld_, part.lp, part.lg);

        // LOCAL AMG preconditioner for the distributed pressure solve (the #1 lever over point-Jacobi, which
        // caps its iteration count and diverges on graded meshes). Built ONCE per rank from this partition's
        // internal-face addressing + face areas (static geometry), exactly like the single-GPU pressure AMG.
        // It intentionally omits the PROCESSOR interface -- as a preconditioner inside the interface-coupled
        // outer CG that is the block-Jacobi/additive-Schwarz design (the outer matvec carries the coupling). A
        // CYCLIC patch, however, is a long-range edge the Galerkin coarse operator cannot represent, so AMG is
        // skipped (fall back to Jacobi-PCG) when the local mesh has one, matching the single-GPU guard.
        {
            const bool amgOff = (std::getenv("BRAE_PARALLEL_AMG") && std::atoi(std::getenv("BRAE_PARALLEL_AMG")) == 0);
            bool hasCyclic = false;
            for (const FvPatch& p : lp) if (p.type == "cyclic" || p.type == "cyclicAMI") hasCyclic = true;
            if (!amgOff && !hasCyclic && lnC_ > 64)
            {
                const std::vector<label> ownerInt(P_.Lm.mesh.owner().begin(), P_.Lm.mesh.owner().begin() + nIf_);
                const std::vector<label> neiInt(P_.Lm.mesh.neighbour());
                std::vector<scalar> fwAMG(P_.lg.magSf().begin(), P_.lg.magSf().begin() + nIf_);   // geometric face area
                const bool tS = std::getenv("BRAE_TIME_STARTUP") != nullptr;
                const auto t0 = std::chrono::high_resolution_clock::now();

                // AMG hierarchy: the agglomeration STRUCTURE is static per (mesh, nProcs), so DISK-CACHE it like the
                // single-GPU buildOrLoadAMG -- a warm launch reloads instead of re-agglomerating. Per-rank cache file
                // (each rank's local partition differs); freshness vs the GLOBAL owner (the local mesh is a
                // deterministic function of it + nProcs). Only the structure is cached; amgGalerkin rebuilds the
                // coarse VALUES each step. Default ON when a cacheDir is given; BRAE_AMG_CACHE=0 forces a rebuild.
                bool loaded = false;
                const bool cacheOn = !amgCacheDir.empty()
                    && !(std::getenv("BRAE_AMG_CACHE") && std::atoi(std::getenv("BRAE_AMG_CACHE")) == 0);
                std::string cacheFile;
                if (cacheOn)
                {
                    cacheFile = amgCacheDir + "/.brae_amg_p" + std::to_string(P_.rank)
                              + "_np" + std::to_string(Pstream::nProcs());
                    namespace fs = std::filesystem;
                    std::error_code ec;
                    const std::string ownerF = amgCacheDir + "/owner";
                    if (fs::exists(cacheFile, ec) && fs::exists(ownerF, ec)
                        && fs::last_write_time(cacheFile, ec) >= fs::last_write_time(ownerF, ec))
                        loaded = loadAMGCache(cacheFile, pAMG_);
                }
                if (!loaded)
                {
                    pAMG_ = buildAMG(ownerInt, neiInt, fwAMG, static_cast<int>(lnC_));
                    if (cacheOn) writeAMGCache(pAMG_, cacheFile);
                }
                useAMG_ = true;
                if (tS && P_.rank == 0)
                    std::printf("[startup] buildAMG      %8.1f ms  (%s, %d levels)\n",
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - t0).count(),
                                loaded ? "cache HIT" : "built", pAMG_.nLevels());

                // ---- STAGE 1 OVERLAP (BRAE_OVERLAP>=1): AMG on OWNED + 1 GHOST RING (owned-only AMG stays pAMG_) ----
                // Extend the local internal graph with one ghost node per interface face, connected to its owner cell
                // by the interface face (coeff = ifCoeff, folded in at assembly time). Ghost of interface i, face f ->
                // extended cell lnC_ + ghOff_[i] + f. The structure is static; the matrix values (owned + exchanged
                // ghost diagonal + interface off-diagonals) are assembled per step. faceWeight for the interface faces
                // is a placeholder 1.0 here (only affects the agglomeration heuristic; refined later).
                if (std::getenv("BRAE_OVERLAP") && std::atoi(std::getenv("BRAE_OVERLAP")) >= 1)
                {
                    std::vector<label>  exOwner(ownerInt), exNei(neiInt);
                    std::vector<scalar> exFw(fwAMG);
                    // CUSTOM AGGREGATION: give the owned<->ghost faces a WEAK aggregation weight so buildAMG keeps each
                    // ghost as its own aggregate instead of lumping it with an owned cell. (Lumping let the ghost's large
                    // neighbour-diagonal + residual contaminate the owned coarse correction -> divergence.) The real
                    // coupling is still captured: amgGalerkin builds the coarse operator from the ACTUAL matrix values,
                    // not these aggregation weights. wGhost = 1e-3 * (smallest internal face weight).
                    scalar wmin = fwAMG.empty() ? 1.0 : fwAMG[0];
                    for (scalar w : fwAMG) wmin = std::fmin(wmin, std::fabs(w));
                    const scalar wGhost = 1e-3 * wmin;
                    ghOff_.assign(P_.Lm.procFaceCells.size() + 1, 0);
                    int nGhost = 0;
                    for (std::size_t i = 0; i < P_.Lm.procFaceCells.size(); ++i)
                    {
                        const std::vector<label>& fc = P_.Lm.procFaceCells[i];
                        ghOff_[i] = nGhost;
                        for (std::size_t f = 0; f < fc.size(); ++f)
                        {
                            exOwner.push_back(fc[f]);
                            exNei.push_back(static_cast<label>(lnC_) + static_cast<label>(nGhost) + static_cast<label>(f));
                            exFw.push_back(wGhost);
                        }
                        nGhost += static_cast<int>(fc.size());
                    }
                    ghOff_[P_.Lm.procFaceCells.size()] = nGhost;
                    nGhost_ = nGhost;
                    if (nGhost > 0)
                    {
                        pAMGext_ = buildAMG(exOwner, exNei, exFw, static_cast<int>(lnC_) + nGhost);
                        // Extended fine matrix: addressing built once (zero values), assembled per step in solvePressure.
                        const std::vector<scalar> z0(static_cast<std::size_t>(lnC_) + nGhost, 0.0);
                        const std::vector<scalar> zf(exOwner.size(), 0.0);
                        exLdu_ = buildDeviceLdu(z0, zf, zf, exOwner, exNei, static_cast<int>(lnC_) + nGhost);
                        exRes_.resize(static_cast<int>(lnC_) + nGhost);
                        exCorr_.resize(static_cast<int>(lnC_) + nGhost);
                        useOverlap_ = true;
                        if (P_.rank == 0)
                            std::printf("[overlap] pAMGext: %d owned + %d ghost = %d ext cells, %d levels\n",
                                        (int)lnC_, nGhost, (int)lnC_ + nGhost, pAMGext_.nLevels());
                    }
                }
            }
        }
    }

    // STAGE DUMP: write every intermediate of iteration `iter`, gathered to the GLOBAL cell ordering,
    // so an np=1 run (no processor faces -> interface terms inactive -> THE reference discretisation) and an
    // np=N run can be diffed stage by stage. The FIRST stage that differs localises the bug, instead of only
    // seeing a blown-up field N iterations later.
    //
    // This is how the LUST interface bug was found: at iteration 2 (the first iteration where the corrections
    // are non-zero at all -- U starts uniform, so grad(U)=0 and every correction is identically 0 at iteration
    // 1) `luCorr_int` was 3.7e-02 off and `luCorr_tot` 1.2e-09 (the linearUpwind interface term restoring it
    // exactly), while `linCorr_tot` was 6.2e+01 off -- one kernel, isolated. Free when unused (early return).
    //
    // Two traps it also exposed, both worth knowing before adding any interface term:
    //   - dump at an iteration where the term is ACTUALLY NON-ZERO (iteration 1 proves nothing here);
    //   - `sumOff` legitimately DIFFERS between np=1 and np>1: at np=1 those cut faces are internal and
    //     deviceRelaxDiag already counts them from the LDU; at np>1 they arrive via the cycSumOff hook. It is
    //     the complement, not a bug -- check `mDiagR` (which must match) instead.
    void setStageDump(const std::string& path, int iter) { dumpPath_ = path; dumpAt_ = iter; }

    // Enable an MRF rotating cell zone (distributed mirror of the single-GPU device setMRF). Builds the per-rank
    // frame data from this rank's LOCAL mesh/geometry/patches (z is the LOCAL zone) and makes the resident flux
    // RELATIVE. Call once after construction, before stepping. deviceMrfCoriolis adds -V*(Omega x U) to the
    // momentum source (step()), deviceMrfApplyFrameFlux makes phi/phiHbyA relative.
    //   Cut-face gating: buildDeviceMRF marks a processor (cut) face relative whenever the LOCAL cell is in the
    // zone, but a cut face is "internal to the zone" only if BOTH cells are -- and the two ranks must agree.
    // Exchange the per-cell zone mask across the cut and zero the frame flux where the REMOTE cell is outside the
    // zone. At np=1 there are no processor faces, so this is a no-op and the result is bit-identical to single-GPU.
    void setMRF(
        const MRFZone& z,
        const std::vector<std::string>& nonRotating = {})
    {
        mrf_ = buildDeviceMRF(z, P_.Lm.mesh, P_.lg, P_.lp, nonRotating);
        if (halo_.nInterfaces() > 0)
        {
            std::vector<scalar> zc(lnC_, 0.0);   // per-cell zone mask (scalar) for the halo exchange
            for (label c : z.cells) zc[c] = 1.0;
            DeviceBuffer<scalar> zoneF;
            zoneF.copyFrom(zc);
            DeviceBuffer<scalar> remoteMask;     // 1 everywhere; processor faces overwritten with the RAW remote mask
            remoteMask.copyFrom(std::vector<scalar>(nBnd_, 1.0));
            // zero interpolation weights -> bvalKernel returns wf*local + (1-wf)*remote = remote (raw 0/1 remote mask
            // at the cut faces; physical boundary faces are not written, so they keep the initial 1.0).
            std::vector<DeviceBuffer<scalar>> zeroW(halo_.nInterfaces());
            for (int i = 0; i < halo_.nInterfaces(); ++i)
                zeroW[i].copyFrom(std::vector<scalar>(static_cast<std::size_t>(halo_.size(i)), 0.0));
            halo_.exchange(zoneF.data());
            halo_.scatterBoundaryValues(zoneF.data(), zeroW, procStart_, remoteMask.data());
            halo_.waitExchange();
            deviceHadamard(mrf_.frameFluxBnd, mrf_.frameFluxBnd, remoteMask);   // cut faces relative only if remote in zone
        }
        deviceMrfApplyFrameFlux(mrf_, +1.0, phiInt_, phiBnd_);   // initial resident phi -> relative (OF createFields)
    }

    // Enable fvOptions (distributed mirror of the single-GPU device setFvOptions). `fo` must already be LOCALISED
    // to THIS rank -- per-cell arrays sized to nCells(), zone cells in local indices (the app maps global->local
    // via cellProcAddr). Only the per-cell source families are supported here: vectorSemiImplicitSource(U)
    // (momSu -> relaxSrc, momSp -> diagonal) and explicitPorositySource (DarcyForchheimer, evaluated each iter);
    // global/reduction sources (meanVelocityForce, actuationDisk) + scalar sources are refused in the app.
    // No fixedValue-p patch (all zeroGradient/cyclic/AMI/empty) -> singular pressure. Set from the app's p-BC scan so
    // the pressure step regularises the interface-coupled singular system (see needRef_). The app pins the level post-hoc.
    void setPressureSingular(bool s) { needRef_ = s; }

    void setFvOptions(const FvOptionsData& fo)
    {
        hasFvoMom_ = fo.hasMomentum;
        if (fo.hasMomentum)
        {
            for (int c = 0; c < 3; ++c) fvoMomSu_[c].copyFrom(fo.momSu[c]);
            if (!fo.momSp.empty()) fvoMomSp_.copyFrom(fo.momSp);
        }
        if (fo.porActive)
        {
            por_.active = true;
            por_.d = fo.porD;
            por_.f = fo.porF;
            por_.cells.copyFrom(fo.porCells);
        }
    }

    // Enable a distributed cyclicAMI interface. `globalAMIs` = buildAMIInterfaces on the GLOBAL mesh; `cellToPart`
    // the decomposition. Builds this rank's DistributedAMI (re-indexed stencil + NVSHMEM gather schedule); the
    // implicit coupling then flows through the momentum + pressure matvecs (deviceParallelAmul with &ami_).
    // WIP: the matrix diagonal (deviceAmiAssembleMomentum/Laplacian), the explicit flux/div/grad/H ops and the
    // rotational path are NOT yet wired -- so this is not yet enabled by the app (cyclicAMI is still refused).
    void setAMI(const std::vector<AMIInterface>& globalAMIs, const std::vector<label>& cellToPart)
    {
        ami_ = buildDistributedAMI(globalAMIs, cellToPart, P_);
        if (ami_.active)
        {
            // initial conservative AMI flux from U (deviceAmiFlux reads U[own] + interp(forwardT.U[nbr]) -> the
            // interpolated neighbour needs the GATHERED extended field for each component). Sets ami_.local.phi.
            DeviceBuffer<scalar> ux, uy, uz;
            distributedAmiGather(ami_, Uk_[0], ux);
            distributedAmiGather(ami_, Uk_[1], uy);
            distributedAmiGather(ami_, Uk_[2], uz);
            deviceAmiFlux(ami_.local, ux, uy, uz);
        }
    }

    // One distributed SIMPLE iteration. U/p/phi are updated in place on the device. Returns the momentum and
    // pressure INITIAL residuals -- OpenFOAM's SIMPLE convergence measure (fvSolution residualControl).
    ParStepResidual step();

    std::vector<vector> U() const
    {
        const std::vector<scalar> ux = Uk_[0].host(), uy = Uk_[1].host(), uz = Uk_[2].host();
        std::vector<vector> out(lnC_);
        for (label c = 0; c < lnC_; ++c) out[c] = vector{ux[c], uy[c], uz[c]};
        return out;
    }
    std::vector<scalar> p() const { return dp_.host(); }

    // max |phi| over this rank's processor faces. The interface deferred corrections (linearUpwind, LUST) are
    // all multiplied by the cut-face flux, so a test on a mesh where this is ~0 cannot see them at all -- the
    // 3D cavity cuts on a symmetry midplane and measures ~3e-14 here. Tests assert on this to prove teeth.
    // Total Krylov iterations (3 momentum BiCGStab + 1 pressure PCG) since construction. A speedup number is
    // uninterpretable without this: if np=1 and np=2 run different iteration counts (the reduction order
    // changes convergence slightly), wall-clock per SIMPLE step measures SOLVER WORK, not parallel efficiency.
    long krylovIters() const { return kIters_; }

    scalar maxProcFlux() const
    {
        const std::vector<scalar> b = phiBnd_.host();
        scalar mx = 0;
        for (std::size_t i = 0; i < procStart_.size(); ++i)
        {
            const label n = static_cast<label>(halo_.size(static_cast<int>(i)));
            for (label f = 0; f < n; ++f)
                mx = std::fmax(mx, std::fabs(b[procStart_[i] + f]));
        }
        return mx;
    }

    // The global field, gathered from every partition (decomposePar's inverse) -- for output/validation.
    std::vector<scalar> reconstructP() const { return reconstructField(P_.Lm.cellProcAddr, dp_.host(), P_.globalNCells); }

    // Turn on the distributed k-epsilon model: k/eps/nut become device-resident, step() then uses
    // nuEff = nu + nut for the momentum and runs parallelDeviceKEpsilonCorrect after the corrector. The three
    // fields carry their OpenFOAM boundary shapes (wall functions etc.). Idempotent per constructed solver.
    void enableTurbulence(const GeometricField<vector>& U0, const GeometricField<scalar>& k0,
                          const GeometricField<scalar>& eps0, const GeometricField<scalar>& nut0,
                          const KEpsilonCoeffs& co, scalar relaxK = 0.7, scalar relaxEps = 0.7)
    {
        turbulent_ = true;  turbModel_ = 0;
        relaxK_ = relaxK; relaxEps_ = relaxEps;
        keCoeffs_ = co;
        k_.copyFrom(k0.internal); eps_.copyFrom(eps0.internal); nut_.copyFrom(nut0.internal);
        dbK_   = buildDeviceBoundary(k0,   P_.lp, P_.lg);
        dbEps_ = buildDeviceBoundary(eps0, P_.lp, P_.lg);
        wall_ = buildDeviceWallData(P_.Lm.mesh, P_.lg, P_.lp, U0);   // proper no-slip wall BCs
        std::vector<label> isW(lnC_, 0);
        for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
            if (P_.lp[pi].type == "wall")
                for (label i = 0; i < P_.lp[pi].size; ++i) isW[P_.lp[pi].faceCells[i]] = 1;
        isWallCell_.copyFrom(isW);
        // per-boundary-FACE wall mask + near-wall distance (deviceBoundaryNut indexes by boundary face, NOT by
        // cell): iterate patches skipping cyclic, exactly as the single-GPU bndIsWall_/bndY_ (device_simple_foam).
        {
            const std::vector<std::vector<scalar>> yW = nearWallDist(P_.Lm.mesh, P_.lg, P_.lp);
            std::vector<label> biw; std::vector<scalar> byy;
            for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
            {
                if (P_.lp[pi].type == "cyclic" || P_.lp[pi].type == "cyclicAMI") continue;
                const bool wall = (P_.lp[pi].type == "wall");
                for (label i = 0; i < P_.lp[pi].size; ++i) { biw.push_back(wall ? 1 : 0); byy.push_back(wall ? yW[pi][i] : 0.0); }
            }
            bndIsWall_.copyFrom(biw);
            bndY_.copyFrom(byy);
        }
        // geomD = magSf*procDelta per interface (static)
        std::size_t pj = 0;
        for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
        {
            if (P_.lp[pi].type != "processor") continue;
            std::vector<scalar> gd(P_.lp[pi].size);
            for (label i = 0; i < P_.lp[pi].size; ++i)
                gd[i] = P_.lg.magSf()[P_.lp[pi].start + i] * P_.procDelta[pj][i];
            geomD_.emplace_back();
            geomD_.back().copyFrom(gd);
            ++pj;
        }
        validateTurbulence();   // OF turbulence->validate(): seed nut from the bounded k/eps BEFORE iteration 1
    }

    // OF simpleFoam.C:92 `turbulence->validate()` -> eddyViscosity::validate() -> correctNut(): bound the
    // fields just read from disk and recompute the INTERNAL nut from them BEFORE iteration 1. The single-GPU
    // path already does this (device_simple_foam validateTurbulence); the DISTRIBUTED path did not, so its
    // first momentum predictor ran with nut straight from 0/nut -- `uniform 0` in pitzDaily and motorBike,
    // i.e. fully laminar. Measured consequence on pitzFS (Re-high, graded): iteration 1 overshoots U by ~108x
    // (2.7e3 vs 2.5e1), the k/epsilon production then sees those velocity gradients, epsilon comes out ~85x
    // too large, nut = Cmu*k^2/eps collapses, and the run is all-NaN by iteration ~10. Seeding nut here is
    // the whole fix; every model needs it, so it is one method called from each enable*.
    void validateTurbulence()
    {
        if (!turbulent_) return;
        if (turbModel_ == 3)                       // SpalartAllmaras: bound(nuTilda,0), nut = nuTilda*fv1
        {
            deviceBoundField(dm_, nuTilda_, 0.0);
            deviceNutSA(nuTilda_, nu_, saCoeffs_.Cv1, nut_);
            return;
        }
        deviceBoundField(dm_, k_,   1e-15);        // bound(k_, kMin_)
        deviceBoundField(dm_, eps_, 1e-15);        // bound(epsilon_|omega_, ...Min_)
        if (turbModel_ == 1 || turbModel_ == 2)    // kOmegaSST(-LM): nut = a1*k/max(a1*omega, b1*F2*sqrt(S2))
        {
            // halo-consistent gradU, exactly as parallelDeviceKOmegaSSTCorrect builds it (cut value from the halo)
            const DeviceBuffer<scalar>* Uc[3] = { &Uk_[0], &Uk_[1], &Uk_[2] };
            DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * lnC_);
            for (int i = 0; i < 3; ++i)
            {
                DeviceBuffer<scalar> ub;
                deviceBCValue(dbU_.comp[i], *Uc[i], ub);
                halo_.exchange(Uc[i]->data());
                halo_.scatterBoundaryValues(Uc[i]->data(), weightsD_, procStart_, ub.data());
                halo_.waitExchange();
                DeviceBuffer<scalar> gx, gy, gz;
                deviceGaussGrad(dm_, *Uc[i], ub, gx, gy, gz);
                const std::size_t n = static_cast<std::size_t>(lnC_);
                cudaCheck(cudaMemcpyAsync(gradU.data() + (0*3+i)*n, gx.data(), n*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "validate gradU x");
                cudaCheck(cudaMemcpyAsync(gradU.data() + (1*3+i)*n, gy.data(), n*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "validate gradU y");
                cudaCheck(cudaMemcpyAsync(gradU.data() + (2*3+i)*n, gz.data(), n*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "validate gradU z");
            }
            DeviceBuffer<scalar> S2, F2;
            deviceS2(gradU, static_cast<int>(lnC_), S2);
            deviceF2(k_, eps_, yCell_, nu_, sstCoeffs_, F2);
            deviceNutSST(k_, eps_, F2, S2, sstCoeffs_, nut_);
        }
        else                                       // kEpsilon: nut = Cmu*k^2/eps
            deviceNut(k_, eps_, nut_, keCoeffs_);
    }

    // Turn on distributed k-omega SST. Second field is OMEGA (stored in eps_/dbEps_), plus the cell wall
    // distance y (F1/F2). Reuses the k-eps setup; keCoeffs_ is derived for the momentum wall nut
    // (nutkWallFunction uses Cmu=betaStar, kappa, E). step() then calls parallelDeviceKOmegaSSTCorrect.
    void enableTurbulenceSST(const GeometricField<vector>& U0, const GeometricField<scalar>& k0,
                             const GeometricField<scalar>& omega0, const GeometricField<scalar>& nut0,
                             const KOmegaSSTCoeffs& co, scalar relaxK = 0.7, scalar relaxOmega = 0.7)
    {
        turbulent_ = true;  turbModel_ = 1;
        relaxK_ = relaxK; relaxEps_ = relaxOmega;
        sstCoeffs_ = co;
        keCoeffs_.Cmu = co.betaStar; keCoeffs_.kappa = co.kappa; keCoeffs_.E = co.E;   // momentum wall nut
        k_.copyFrom(k0.internal); eps_.copyFrom(omega0.internal); nut_.copyFrom(nut0.internal);
        dbK_   = buildDeviceBoundary(k0,     P_.lp, P_.lg);
        dbEps_ = buildDeviceBoundary(omega0, P_.lp, P_.lg);
        wall_ = buildDeviceWallData(P_.Lm.mesh, P_.lg, P_.lp, U0);
        yCell_.copyFrom(cellWallDist(P_.Lm.mesh, P_.lg, P_.lp));
        std::vector<label> isW(lnC_, 0);
        for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
            if (P_.lp[pi].type == "wall")
                for (label i = 0; i < P_.lp[pi].size; ++i) isW[P_.lp[pi].faceCells[i]] = 1;
        isWallCell_.copyFrom(isW);
        {
            const std::vector<std::vector<scalar>> yW = nearWallDist(P_.Lm.mesh, P_.lg, P_.lp);
            std::vector<label> biw; std::vector<scalar> byy;
            for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
            {
                if (P_.lp[pi].type == "cyclic" || P_.lp[pi].type == "cyclicAMI") continue;
                const bool wall = (P_.lp[pi].type == "wall");
                for (label i = 0; i < P_.lp[pi].size; ++i) { biw.push_back(wall ? 1 : 0); byy.push_back(wall ? yW[pi][i] : 0.0); }
            }
            bndIsWall_.copyFrom(biw);
            bndY_.copyFrom(byy);
        }
        std::size_t pj = 0;
        for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
        {
            if (P_.lp[pi].type != "processor") continue;
            std::vector<scalar> gd(P_.lp[pi].size);
            for (label i = 0; i < P_.lp[pi].size; ++i)
                gd[i] = P_.lg.magSf()[P_.lp[pi].start + i] * P_.procDelta[pj][i];
            geomD_.emplace_back();
            geomD_.back().copyFrom(gd);
            ++pj;
        }
        validateTurbulence();   // OF turbulence->validate(): seed nut from the bounded k/omega BEFORE iteration 1
    }
    // kOmegaSSTLM (Langtry-Menter transition): the shared SST scaffolding + the two extra transition transport
    // fields ReThetat/gammaInt (device-resident, with boundaries). step() then runs parallelDeviceKOmegaSSTLMCorrect.
    void enableTurbulenceSSTLM(const GeometricField<vector>& U0, const GeometricField<scalar>& k0,
                               const GeometricField<scalar>& omega0, const GeometricField<scalar>& nut0,
                               const GeometricField<scalar>& ReThetat0, const GeometricField<scalar>& gammaInt0,
                               const KOmegaSSTCoeffs& co, scalar relaxK = 0.7, scalar relaxOmega = 0.7)
    {
        enableTurbulenceSST(U0, k0, omega0, nut0, co, relaxK, relaxOmega);
        turbModel_ = 2;
        ReThetat_.copyFrom(ReThetat0.internal);
        gammaInt_.copyFrom(gammaInt0.internal);
        dbReThetat_ = buildDeviceBoundary(ReThetat0, P_.lp, P_.lg);
        dbGammaInt_ = buildDeviceBoundary(gammaInt0, P_.lp, P_.lg);
    }
    // SpalartAllmaras (one-equation nuTilda): single transport field + nut = nuTilda*fv1. The momentum wall nut
    // uses Spalding (deviceBoundaryNutSpalding) in step(), so build the per-boundary wall flag + distance too.
    void enableTurbulenceSA(const GeometricField<vector>& U0, const GeometricField<scalar>& nuTilda0,
                            const GeometricField<scalar>& nut0, const SpalartAllmarasCoeffs& co, scalar relaxNuTilda = 0.7)
    {
        turbulent_ = true;  turbModel_ = 3;
        relaxK_ = relaxNuTilda; relaxEps_ = relaxNuTilda;
        saCoeffs_ = co;
        nuTilda_.copyFrom(nuTilda0.internal);
        nut_.copyFrom(nut0.internal);
        dbNuTilda_ = buildDeviceBoundary(nuTilda0, P_.lp, P_.lg);
        wall_ = buildDeviceWallData(P_.Lm.mesh, P_.lg, P_.lp, U0);
        yCell_.copyFrom(cellWallDist(P_.Lm.mesh, P_.lg, P_.lp));
        {
            const std::vector<std::vector<scalar>> yW = nearWallDist(P_.Lm.mesh, P_.lg, P_.lp);
            std::vector<label> biw; std::vector<scalar> byy;
            for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
            {
                if (P_.lp[pi].type == "cyclic" || P_.lp[pi].type == "cyclicAMI") continue;
                const bool wall = (P_.lp[pi].type == "wall");
                for (label i = 0; i < P_.lp[pi].size; ++i) { biw.push_back(wall ? 1 : 0); byy.push_back(wall ? yW[pi][i] : 0.0); }
            }
            bndIsWall_.copyFrom(biw);
            bndY_.copyFrom(byy);
        }
        std::size_t pj = 0;   // geomD_ (magSf*procDelta per interface) for the nuTilda faceGammaGeo
        for (std::size_t pi = 0; pi < P_.lp.size(); ++pi)
        {
            if (P_.lp[pi].type != "processor") continue;
            std::vector<scalar> gd(P_.lp[pi].size);
            for (label i = 0; i < P_.lp[pi].size; ++i)
                gd[i] = P_.lg.magSf()[P_.lp[pi].start + i] * P_.procDelta[pj][i];
            geomD_.emplace_back();
            geomD_.back().copyFrom(gd);
            ++pj;
        }
        validateTurbulence();   // OF turbulence->validate(): seed nut = nuTilda*fv1 BEFORE iteration 1
    }
    std::vector<scalar> reconstructNuTilda() const { return reconstructField(P_.Lm.cellProcAddr, nuTilda_.host(), P_.globalNCells); }
    std::vector<scalar> reconstructReThetat() const { return reconstructField(P_.Lm.cellProcAddr, ReThetat_.host(), P_.globalNCells); }
    std::vector<scalar> reconstructGammaInt() const { return reconstructField(P_.Lm.cellProcAddr, gammaInt_.host(), P_.globalNCells); }
    std::vector<scalar> reconstructK()   const { return reconstructField(P_.Lm.cellProcAddr, k_.host(),   P_.globalNCells); }
    std::vector<scalar> reconstructEps() const { return reconstructField(P_.Lm.cellProcAddr, eps_.host(), P_.globalNCells); }
    std::vector<scalar> reconstructNut() const { return reconstructField(P_.Lm.cellProcAddr, nut_.host(), P_.globalNCells); }
    std::vector<scalar> kLocal()   const { return k_.host(); }
    std::vector<scalar> epsLocal() const { return eps_.host(); }
    std::vector<scalar> nutLocal() const { return nut_.host(); }

    // gather `d` to the global ordering and append "<name> v0 v1 ..." to the dump file (master only)
    void dumpStage(const char* name, const DeviceBuffer<scalar>& d, int k = -1) const
    {
        if (dumpPath_.empty() || iter_ != dumpAt_) return;
        cudaStreamSynchronize(cudaStreamPerThread);
        std::vector<scalar> g(P_.globalNCells, 0.0);
        const std::vector<scalar> h = d.host();
        for (label c = 0; c < lnC_ && c < static_cast<label>(h.size()); ++c)
            g[P_.Lm.cellProcAddr[c]] = h[c];
        Pstream::allReduce(g.data(), static_cast<int>(g.size()), ReduceOp::Sum);
        if (!Pstream::master()) return;
        std::ofstream os(dumpPath_, std::ios::app);
        os.precision(17);
        os << name;
        if (k >= 0) os << '_' << k;
        for (scalar v : g) os << ' ' << v;
        os << '\n';
    }

private:
    static std::vector<int> intNbrs(const Partition& part)
    {
        std::vector<int> n;
        for (int q : part.Lm.procNbr) n.push_back(q);
        return n;
    }

    const Partition& P_;
    scalar nu_, relaxU_, relaxP_, tolU_, tolP_, relTolU_, relTolP_, relTolKE_;
    int    maxIter_;
    bool   bounded_ = false;
    bool   linearUpwind_ = false;
    bool   lust_ = false;
    bool   linUpwindV_ = false;   // linearUpwindV: vector-limited internal linearUpwind correction (cut faces plain)
    bool   nonOrth_ = false;
    bool   consistent_ = false;   // SIMPLEC: use rAtU=1/(1/rAU - H1) in the pEqn + the flux/HbyA consistency corrections
    bool   limitedU_ = false;     // div(phi,U) limitedLinear (magSqr): IMPLICIT limited convection, ONE limiter/face on |U|^2
    scalar twoByk_ = 2.0;         // limitedLinear coefficient: 2/max(k_,SMALL) (k_=1 -> 2); limiter = clamp(twoByk*r,0,1)
    bool   limitedV_ = false;     // div(phi,U) limitedLinearV (NVDVTVDV): IMPLICIT limited convection, vector limiter/face
    DeviceMRF mrf_;               // optional rotating cell zone (MRF): Coriolis source + relative flux (inactive by default)
    DistributedAMI ami_;          // optional cyclicAMI interface (implicit coupling in the matvec); inactive by default
    bool   needRef_ = false;      // no fixedValue-p patch -> SINGULAR all-Neumann pressure. With an AMI/cyclic interface
                                  // the interface-coupled singular system needs a symmetric diagonal shift to make the CG
                                  // well-posed (the level is pinned post-hoc in the app). Plain singular cases converge
                                  // without it (verified 3D cavity), so the shift is applied ONLY when ami_.active.
    DeviceBuffer<scalar> amiSumOff_, amiExt_;   // AMI diagonal-dominance term + a gather scratch (explicit ops)
    bool   hasFvoMom_ = false;    // fvOptions vectorSemiImplicitSource(U): per-cell explicit Su + implicit Sp
    DeviceBuffer<scalar> fvoMomSu_[3], fvoMomSp_;   // Su*V per component -> relaxSrc; Sp*V -> momentum diagonal
    DevicePorosity por_;          // fvOptions explicitPorositySource (DarcyForchheimer), evaluated each iter
    std::string dumpPath_;
    int    dumpAt_ = -1;
    mutable int iter_ = 0;
    mutable long kIters_ = 0;
    label  lnC_, nIf_, nBnd_ = 0;
    bool   validC_[3] = { true, true, true };
    DeviceMesh           dm_;
    DeviceVectorBoundary dbU_;
    DeviceBoundary       dbP_, dbRAU_;
    GeometricField<scalar> rAUfld_;
    DeviceHalo           halo_;
    bool     useAMG_ = false;   // pressure preconditioned by the local per-rank AMG V-cycle (else point-Jacobi)
    AMGData  pAMG_;             // the local pressure-Laplacian hierarchy (built once, Galerkin-recoarsened per step)
    // Stage-1 OVERLAP (BRAE_OVERLAP): a SECOND AMG hierarchy on the owned cells + one ghost ring (the neighbour's
    // interface cells), so the local solve sees across the processor cut -> overlapping additive Schwarz. Structure
    // is static (built once here); the extended matrix VALUES are assembled per step. Default off.
    AMGData  pAMGext_;
    bool     useOverlap_ = false;
    int      nGhost_ = 0;
    std::vector<int> ghOff_;    // ghost-block offset per interface: ghost of interface i, face f -> lnC_ + ghOff_[i] + f
    DeviceLduMatrix      exLdu_;         // extended fine matrix (owned + ghost); addressing built once, values per step
    DeviceBuffer<scalar> exRes_, exCorr_;// RAS scratch, length lnC_ + nGhost_
    // Persistent pressure-solve buffers (graph-referenced): STABLE addresses across steps (resize is a no-op when
    // the size is unchanged), so the whole-loop conditional graph (BRAE_PARALLEL_GRAPH=2) captures ONCE and
    // replays across all steps instead of re-instantiating per step. Values are overwritten by the assembly each
    // step (matrix, rhs, interface coeffs, and the psi seed) -- the graph references the buffers, read at replay.
    DeviceBuffer<scalar> pUp_, pLp_, pDiagp_, pbp_, pSolp_;
    std::vector<DeviceBuffer<scalar>> pIfCoeffp_;
    // Per-U-component momentum-BiCGStab whole-loop graph caches (BRAE_PARALLEL_GRAPH=3), keyed on Uk_[k]'s address.
    // unique_ptr so the cache (which owns a CUDA graph + persistent Krylov/matrix buffers) stays non-copyable/stable.
    std::unique_ptr<BiCGGraphCache> bicgGCache_[3];
    std::vector<label>   procStart_;
    std::vector<DeviceBuffer<label>>  faceCellsD_;
    std::vector<DeviceBuffer<scalar>> weightsD_;
    std::vector<DeviceBuffer<scalar>> dOwnD_, dNeiD_;   // linearUpwind: static per-cut-face offsets
    // non-orth ("corrected") laplacian: static per-cut-face geometry + the per-interface face areas
    std::vector<std::vector<scalar>> procNonOrth_;      // host: the IMPLICIT coeff (replaces procDelta)
    std::vector<DeviceBuffer<scalar>> corrVecD_, magSfD_, nuFaceD_;
    // Phase 4b turbulence (k-epsilon): held device-resident like U/p/phi; nut feeds nuEff each step, and
    // correct() runs after the corrector. Laminar path is untouched (turbulent_ = false).
    bool turbulent_ = false;
    scalar relaxK_ = 0.7, relaxEps_ = 0.7;
    DeviceBuffer<scalar> k_, eps_, nut_;
    DeviceBuffer<scalar> ReThetat_, gammaInt_;   // kOmegaSSTLM (turbModel_==2) transition transport fields
    DeviceBuffer<scalar> nuTilda_;               // SpalartAllmaras (turbModel_==3) single transport field
    DeviceWallData wall_;
    DeviceBoundary dbK_, dbEps_;
    DeviceBoundary dbReThetat_, dbGammaInt_;     // kOmegaSSTLM transition-field boundaries
    DeviceBoundary dbNuTilda_;                   // SpalartAllmaras nuTilda boundary
    SpalartAllmarasCoeffs saCoeffs_;             // SpalartAllmaras coefficients
    DeviceBuffer<label> isWallCell_;
    DeviceBuffer<label> bndIsWall_;   // per-boundary-face wall mask (for deviceBoundaryNut)
    DeviceBuffer<scalar> bndY_;       // per-boundary-face near-wall distance
    std::vector<DeviceBuffer<scalar>> geomD_;   // magSf*procDelta per interface (for the k/eps interface gamma)
    // magSf * (nonOrth ? procNonOrth : procDelta) per processor interface, static (nonOrth_ is fixed at ctor). The
    // device geometry factor of the momentum + pEqn interface coefficients, so those coeffs build on-device from the
    // halo-scattered cut field instead of a per-step .host() round-trip (Phase A: device-resident assembly).
    std::vector<DeviceBuffer<scalar>> geomIf_;
    KEpsilonCoeffs keCoeffs_;
    int  turbModel_ = 0;             // 0 = k-epsilon, 1 = k-omega SST
    KOmegaSSTCoeffs sstCoeffs_;
    DeviceBuffer<scalar> yCell_;    // cell wall distance (SST F1/F2)
    DeviceBuffer<scalar> Uk_[3], dp_, phiInt_, phiBnd_;
    DeviceBuffer<scalar> ones_, zeroSrc_, zeroBnd_, nuEffBnd_, nuCell_;
};

// Distributed H() per component: the local deviceMatrixH plus the processor-interface term, i.e. the device
// counterpart of host parallelMatrixH (H = (diag*psi - A_parallel*psi + source)/V). Exchanges psiK itself, so
// the caller does not need to pre-exchange. `faceCells[i]`/`ifCoeff[i]` are interface i's addressing/coeffs.
inline void deviceParallelMatrixH(
    const DeviceLduView& A,
    const DeviceMesh& dm,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& ifCoeff,
    const DeviceBuffer<scalar>& psiK,
    const DeviceBuffer<scalar>& sourceK,
    const DeviceBuffer<scalar>& bdDiagK,
    const DeviceBuffer<scalar>& bdSrcK,
    DeviceBuffer<scalar>& Hk,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    halo.postExchange(psiK.data(), stream);                                   // psiNbr for the interface term
    deviceMatrixH(A, dm, psiK, sourceK, bdDiagK, bdSrcK, Hk);                 // local H (overlaps the transfer)
    halo.waitExchange(stream);
    for (int i = 0; i < halo.nInterfaces(); ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::matrixHInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            ifCoeff[i].data(),
            halo.recvData(i),
            dm.V.data(),
            Hk.data(),
            n);
    }
    // REQUIRED: the interface kernels above READ the shared recv buffer. Without this barrier a neighbour's
    // NEXT exchange (e.g. the following velocity component's H) can overwrite our recv buffer while these
    // kernels are still reading it -- see the hazard note in device_halo.cuh. Calling H() per component in a
    // loop hits this immediately.
    halo.waitExchange(stream);
}

// One distributed SIMPLE iteration -- the sequence validated stage-by-stage against host
// parallelSimpleStepLaminar in test_gpu_parallel_predictor.
// parallelDeviceKEpsilonCorrect needs the interface helpers (deviceMomentumInterface, ...) defined
// above; include it here, AFTER them, to break the simple<->turbulence include cycle.
} // (reopened below)
#include "parallel_device_turbulence.cuh"
namespace brae {
inline ParStepResidual ParallelDeviceSimple::step()
{
    ParStepResidual res;
    ++iter_;

    // ---- effective viscosity for this step: nu (laminar) or nu + nut (turbulent, refreshed from the model) ----
    // Laminar: nuEffCell = nuCell_ (=nu), nuEffBndEff = nuEffBnd_ (=nu), nuEff_f = nu, nuEffScatBnd = nu-at-cut.
    DeviceBuffer<scalar> nuEffCell, nuEffBndEff, nuEff_f, nuEffScatBnd;
    if (turbulent_)
    {
        deviceCopy(nuEffCell, nut_);   deviceAxpy(1.0, nuCell_, nuEffCell);     // nu + nut (cell)
        deviceInterpolate(dm_, nuEffCell, nuEff_f);                            // internal faces
        // boundary faces: nu + the TRUE wall nut. kEps/SST use nutkWallFunction; SpalartAllmaras uses Spalding.
        DeviceBuffer<scalar> nutBnd;
        if (turbModel_ == 3)
            deviceBoundaryNutSpalding(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], nut_, nu_, saCoeffs_, nutBnd);
        else
            deviceBoundaryNut(dbU_.comp[0], bndIsWall_, bndY_, k_, nut_, nu_, nutBnd, keCoeffs_);
        deviceCopy(nuEffBndEff, nutBnd);   deviceAxpy(1.0, nuEffBnd_, nuEffBndEff);   // + nu everywhere
    }
    else
    {
        nuEff_f.copyFrom(std::vector<scalar>(nIf_, nu_));
        deviceCopy(nuEffCell, nuCell_);
        deviceCopy(nuEffBndEff, nuEffBnd_);
    }
    // nuEff at the cut faces: the halo-interpolated cell value (nu on both sides for laminar -> unchanged)
    nuEffScatBnd.copyFrom(std::vector<scalar>(nBnd_, 0.0));
    halo_.exchange(nuEffCell.data());
    halo_.scatterBoundaryValues(nuEffCell.data(), weightsD_, procStart_, nuEffScatBnd.data());
    halo_.waitExchange();

    // limitedLinear (magSqr): build the limited convection IMPLICITLY (like OF -- the limited weights go INTO the
    // matrix, not a deferred source). OF's NVDTVD::r takes a scalar, so reduce U to s=|U|^2, form grad(s) by Gauss,
    // and deviceDivLimitedCoeffs computes ONE limiter per face applied to all 3 components (the convection matrix is
    // shared, so it CAN be implicit). A deferred (divLim-divUp).U source instead UNDER-converges here: unlike
    // linearUpwind's mild grad correction (which deferred handles to 0.6% of OF), limitedLinear blends toward
    // central, and an upwind matrix + explicit central push settles near upwind (measured ~8% off OF). The limiter
    // is lagged on U (Picard), exactly as OF does. INTERNAL faces only -> exact at np=1; the cut stays upwind (np>1).
    // Both limitedLinear (magSqr) and limitedLinearV (NVDVTVDV) need grad(U_k); magSqr additionally needs
    // s=|U|^2 and grad(s). limGU{x,y,z}[k] = grad(U_k); for magSqr, limS/limG{x,y,z} = s and grad(s).
    DeviceBuffer<scalar> limS, limGx, limGy, limGz, limGUx[3], limGUy[3], limGUz[3];
    if (limitedU_ || limitedV_)
    {
        DeviceBuffer<scalar> ub[3];
        for (int k = 0; k < 3; ++k)
        {
            deviceBCValue(dbU_.comp[k], Uk_[k], ub[k]);
            halo_.exchange(Uk_[k].data());
            halo_.scatterBoundaryValues(Uk_[k].data(), weightsD_, procStart_, ub[k].data());
            halo_.waitExchange();
            deviceGaussGrad(dm_, Uk_[k], ub[k], limGUx[k], limGUy[k], limGUz[k]);
        }
        if (limitedU_)   // magSqr: s = |U|^2 (cell + boundary) and grad(s) by Gauss (== OF fvc::grad(magSqr(U)))
        {
            DeviceBuffer<scalar> sb, t;
            deviceHadamard(limS, Uk_[0], Uk_[0]);
            deviceHadamard(t, Uk_[1], Uk_[1]); deviceAxpy(1.0, t, limS);
            deviceHadamard(t, Uk_[2], Uk_[2]); deviceAxpy(1.0, t, limS);
            deviceHadamard(sb, ub[0], ub[0]);
            deviceHadamard(t, ub[1], ub[1]); deviceAxpy(1.0, t, sb);
            deviceHadamard(t, ub[2], ub[2]); deviceAxpy(1.0, t, sb);
            deviceGaussGrad(dm_, limS, sb, limGx, limGy, limGz);
        }
    }

    // ---- momentum matrix: div(phi,U) - laplacian(nuEff,U), + processor interface ----
    DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
    if (limitedV_)        // NVDVTVDV vector limiter -> implicit limited convection (one limiter/face, all 3 components)
        deviceDivLimitedVCoeffs(dm_, phiInt_, Uk_, limGUx, limGUy, limGUz, twoByk_, mDiag, mUp, mLo);
    else if (limitedU_)   // magSqr single limiter -> implicit limited convection (weights W shared across components)
        deviceDivLimitedCoeffs(dm_, phiInt_, limS, limGx, limGy, limGz, twoByk_, mDiag, mUp, mLo);
    else
        deviceDivUpwindCoeffs(dm_, phiInt_, mDiag, mUp, mLo);
    deviceLaplacianCoeffs(dm_, nuEff_f, lD, lU, lL, nonOrth_);
    deviceAxpy(-1.0, lD, mDiag);
    deviceAxpy(-1.0, lU, mUp);
    deviceAxpy(-1.0, lL, mLo);

    // cyclicAMI momentum coupling: assemble the interface off-diagonal into mDiag (ifCoeff = -(nuFace*dc*magSf) +
    // min(phi,0); diag[own] += (nuFace*dc*magSf) + max(phi,0)) from ami_.phi + the GATHERED nuEff, and the
    // diagonal-dominance term amiSumOff_ (added to sumOff before deviceRelaxDiag). Rotation is inside the op (fT).
    if (ami_.active)
    {
        distributedAmiGather(ami_, nuEffCell, amiExt_);
        deviceAmiAssembleMomentum(ami_.local, amiExt_, mDiag);
        amiSumOff_.copyFrom(std::vector<scalar>(static_cast<std::size_t>(lnC_), 0.0));
        deviceAmiOffDiagSum(ami_.local, amiSumOff_);
        if (ami_.local.rotational) deviceAmiScaleImplicit(ami_.local);   // per-comp ifCoeffC = ifCoeff*forwardT[kk][kk]
    }

    // fvOptions vectorSemiImplicitSource(U) implicit Sp: diag -= Sp*V (== fvm::Sp(Sp,U)); the explicit Su feeds
    // relaxSrc below. explicitPorositySource: DarcyForchheimer isotropic resistance into the diagonal (from |U|).
    if (fvoMomSp_.size()) deviceAxpy(-1.0, fvoMomSp_, mDiag);
    if (por_.active) deviceFvoPorosityDiag(por_, nu_, dm_.V, Uk_[0], Uk_[1], Uk_[2], mDiag);

    // bounded Gauss upwind: - fvm::Sp(fvc::div(phi), U). Diagonal gets -V*div(phi); it stabilises the
    // transient (a rest-start cell where div(phi)!=0 at low nu blows up otherwise) and vanishes at
    // convergence. Mirrors host parallelSimpleStepLaminar's `bounded` branch.
    if (bounded_)
    {
        DeviceBuffer<scalar> divPhi, sp;
        deviceDiv(dm_, phiInt_, phiBnd_, divPhi);
        // div(phi) MUST sum ALL faces (OF fvc::div): the AMI-face flux too. Omitting it makes the AMI cells see the
        // un-cancelled interface flux as a spurious divergence, so the bounded term injects an asymmetric diagonal ->
        // asymmetric rAU/HbyA -> the rotational SIMPLE loop limit-cycles. Mirrors the single-GPU bounded branch.
        if (ami_.active) deviceAmiAddDiv(ami_.local, dm_.V, divPhi);
        deviceHadamard(sp, dm_.V, divPhi);
        deviceAxpy(-1.0, sp, mDiag);
    }

    // Processor-interface coupling, ON-DEVICE (Phase A): phiF[i] = the cut-face flux (a device slice of phiBnd_),
    // coeffGeo[i] = nuEff-at-cut * geomIf_[i] via interfaceCoeffKernel. Both were a per-step .host() round-trip of
    // phiBnd_ and nuEffScatBnd + a host multiply; now the whole coefficient stays on the device.
    const int nIface = halo_.nInterfaces();
    std::vector<DeviceBuffer<scalar>> phiF(nIface), coeffGeo(nIface);
    {
        constexpr int TPB = 128;
        for (int i = 0; i < nIface; ++i)
        {
            const int n = static_cast<int>(halo_.size(i));
            if (n <= 0) continue;
            phiF[i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(phiF[i].data(), phiBnd_.data() + procStart_[i], n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread),
                "phiF cut slice");
            coeffGeo[i].resize(n);
            detail::interfaceCoeffKernel<<<(n + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
                coeffGeo[i].data(), nuEffScatBnd.data(), static_cast<int>(procStart_[i]), geomIf_[i].data(), n);
        }
    }
    std::vector<DeviceBuffer<scalar>> ifCoeff;
    deviceMomentumInterface(halo_, faceCellsD_, phiF, coeffGeo, mDiag, ifCoeff);
    dumpStage("mDiag", mDiag);

    // ---- relax (the processor interface counts toward diagonal dominance) ----
    DeviceBuffer<scalar> iC[3], bC[3], relaxSrc[3];
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> lIC, lBC;
        deviceBCDivCoeffs(dbU_.comp[k], phiBnd_, iC[k], bC[k]);
        deviceBCLaplacianCoeffsFace(dbU_.comp[k], nuEffBndEff, lIC, lBC);
        deviceAxpy(-1.0, lIC, iC[k]);
        deviceAxpy(-1.0, lBC, bC[k]);
    }
    DeviceBuffer<scalar> sumOff(std::vector<scalar>(lnC_, 0.0));
    deviceInterfaceOffDiagSum(halo_, faceCellsD_, ifCoeff, sumOff);
    if (ami_.active) deviceAxpy(1.0, amiSumOff_, sumOff);   // AMI off-diagonals count toward diagonal dominance too
    DeviceBuffer<scalar> mDiagR, delta;
    deviceRelaxDiag(deviceLduView(dm_, mDiag, mUp, mLo), dm_, iC[0], relaxU_, mDiagR, delta, sumOff.data());
    dumpStage("sumOff", sumOff);
    dumpStage("mDiagR", mDiagR);
    dumpStage("delta", delta);

    // The explicit stress source, V*fvc::div(nuEff*dev2(T(grad U))), from the PRE-predictor U -- processor
    // faces coupled via the halo. Host parallelSimpleStepLaminar folds this into Ml.source BEFORE relax, so
    // it must sit in the same source relax then adds delta*U_old to, i.e. the one H() reads later.
    DeviceProcStress proc;
    proc.halo      = &halo_;
    proc.weights   = &weightsD_;
    proc.procStart = &procStart_;
    DeviceBuffer<scalar> sX, sY, sZ;
    // divDevReff stress. With cyclicAMI, the gradU at the interface reads the AMI-interpolated neighbour U -- which
    // after decomposition is remote -- so pass the GATHERED extended U per component (its [0..nLocal) is the local
    // field the internal Gauss grad + halo use; nbrCell reads into the gathered tail). deviceDivDevReff then does the
    // AMI grad internally (deviceAmiInterpolateVec + deviceAmiAddGradRot rotational / deviceAmiAddGrad translational).
    if (ami_.active)
    {
        DeviceBuffer<scalar> uxE, uyE, uzE;
        distributedAmiGather(ami_, Uk_[0], uxE);
        distributedAmiGather(ami_, Uk_[1], uyE);
        distributedAmiGather(ami_, Uk_[2], uzE);
        deviceDivDevReff(dm_, dbU_, uxE, uyE, uzE, nuEffCell, nuEffBndEff, sX, sY, sZ, nullptr, &ami_.local, &proc);
    }
    else
        deviceDivDevReff(dm_, dbU_, Uk_[0], Uk_[1], Uk_[2], nuEffCell, nuEffBndEff, sX, sY, sZ, nullptr, nullptr, &proc);
    DeviceBuffer<scalar>* sS[3] = { &sX, &sY, &sZ };
    dumpStage("stress", sX, 0);
    dumpStage("stress", sY, 1);
    dumpStage("stress", sZ, 2);
    // ROTATIONAL AMI deferred-rotation split: the cross-component rotation Omega->own can't be fully implicit in the
    // segregated per-component solve. The matrix carries the diagonal part (ifCoeffC = ifCoeff*forwardT[kk][kk], from
    // deviceAmiScaleImplicit), and the off-diagonal mixing goes to the source via the UN-rotated AMI-interp of U_old
    // (amiUint, deviceAmiInterpolate); the full-rotation H (deviceAmiAddH above) then closes it consistently.
    DeviceBuffer<scalar> amiUint[3];
    if (ami_.active && ami_.local.rotational)
        for (int l = 0; l < 3; ++l)
        {
            DeviceBuffer<scalar> ue;
            distributedAmiGather(ami_, Uk_[l], ue);
            deviceAmiInterpolate(ami_.local, ue, amiUint[l]);
        }
    for (int k = 0; k < 3; ++k)
    {
        deviceHadamard(relaxSrc[k], delta, Uk_[k]);      // delta*U_old
        deviceAxpy(1.0, *sS[k], relaxSrc[k]);            // + V*divSig  -> == host Ml.source
        if (mrf_.active)                                 // MRF Coriolis: source -= V*(Omega x U)_k on zone cells
            deviceMrfCoriolis(mrf_, dm_.V, Uk_[0], Uk_[1], Uk_[2], k, relaxSrc[k]);
        if (hasFvoMom_)                                  // fvOptions explicit momentum source Su*V (== fvOptions(U))
            deviceAxpy(1.0, fvoMomSu_[k], relaxSrc[k]);
        if (por_.active)                                 // porosity anisotropic remainder into the source
            deviceFvoPorositySource(por_, k, nu_, dm_.V, Uk_[0], Uk_[1], Uk_[2], relaxSrc[k]);
        // NOTE: the deferred off-diagonal rotation mixing is NOT added here -- relaxSrc feeds BOTH the predictor
        // AND UEqn.H(), but the deferred split is a PREDICTOR-ONLY term (H uses the full-rotation deviceAmiAddH on
        // the post-solve U). It is added to the per-component predictor source `s` in the momentum loop below,
        // matching the single-GPU (device_simple_foam: deviceAmiAddDeferredRot into `s`, not relaxSrc).
    }

    // linearUpwind: the matrix stays upwind and this deferred correction is an explicit source. It must land
    // in relaxSrc -- the SAME source H() reads later -- not only in the predictor's rhs, or HbyA would be
    // built from an upwind-only source and the corrector would disagree with the predictor.
    // LUST also runs the linearUpwind correction (OF LUST.H = 0.75*linear + 0.25*linearUpwind), mirroring the
    // single-GPU `if (ctl_.linearUpwind || ctl_.lust)`.
    if (linearUpwind_ || lust_)
    {
        // grad(U_k) for all 3 components (halo-consistent), computed TOGETHER because linearUpwindV's vector limiter
        // couples the components. gUx[k]=dU_k/dx etc. Then the internal deferred correction is either the plain
        // per-component linearUpwind or (linearUpwindV) the vector-limited deviceLinearUpwindVCorr computed once.
        // (limitedLinear does NOT come here -- it is built implicitly into the matrix above.)
        DeviceBuffer<scalar> gUx[3], gUy[3], gUz[3];
        // AMI-face contribution to grad(U): the single-GPU BAKES this into gU (deviceAmiAddGradRot/Grad) so the
        // INTERNAL linearUpwind reconstruction at AMI-adjacent cells carries the interface neighbour. Without it the
        // reconstruction is wrong there -> the deferred source diverges (the AMI-FACE deviceAmiAddLinUpwindCorr below
        // is a SEPARATE term for the AMI faces themselves, NOT a substitute). The non-orth block gets away without it
        // because deviceAmiAddLapCorr recomputes the interface term from the gathered gradients.
        DeviceBuffer<scalar> amiURot[3];   // rotated AMI-interp of the current U (for the rotational grad correction)
        if (ami_.active && ami_.local.rotational)
        {
            DeviceBuffer<scalar> uxE, uyE, uzE;
            distributedAmiGather(ami_, Uk_[0], uxE);
            distributedAmiGather(ami_, Uk_[1], uyE);
            distributedAmiGather(ami_, Uk_[2], uzE);
            deviceAmiInterpolateVec(ami_.local, uxE, uyE, uzE, amiURot[0], amiURot[1], amiURot[2]);
        }
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> ub;
            deviceBCValue(dbU_.comp[k], Uk_[k], ub);
            halo_.exchange(Uk_[k].data());
            halo_.scatterBoundaryValues(Uk_[k].data(), weightsD_, procStart_, ub.data());
            halo_.waitExchange();
            deviceGaussGrad(dm_, Uk_[k], ub, gUx[k], gUy[k], gUz[k]);
            if (ami_.active)
            {
                if (ami_.local.rotational)
                    deviceAmiAddGradRot(ami_.local, Uk_[k], amiURot[k], dm_.V, gUx[k], gUy[k], gUz[k]);
                else
                {
                    DeviceBuffer<scalar> ukE;
                    distributedAmiGather(ami_, Uk_[k], ukE);
                    deviceAmiAddGrad(ami_.local, ukE, dm_.V, gUx[k], gUy[k], gUz[k]);
                }
            }
        }
        // AMI linearUpwind correction reads the rotated neighbour stencil of ALL 3 components' gradients -> gather
        // each of the 9 gradient arrays (extended [local|gathered]). gUE holds the extended gradients per component.
        DeviceBuffer<scalar> gUxE[3], gUyE[3], gUzE[3];
        if (ami_.active)
            for (int l = 0; l < 3; ++l)
            {
                distributedAmiGather(ami_, gUx[l], gUxE[l]);
                distributedAmiGather(ami_, gUy[l], gUyE[l]);
                distributedAmiGather(ami_, gUz[l], gUzE[l]);
            }
        DeviceBuffer<scalar> corrV[3];
        if (linUpwindV_)   // vector-limited internal correction, all 3 at once (cut faces still use the plain grad)
            deviceLinearUpwindVCorr(dm_, phiInt_, gUx, gUy, gUz, Uk_[0], Uk_[1], Uk_[2], corrV[0], corrV[1], corrV[2]);
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> corr;
            if (linUpwindV_) corr = std::move(corrV[k]);
            else             deviceLinearUpwindCorr(dm_, phiInt_, gUx[k], gUy[k], gUz[k], corr);   // internal faces
            deviceLinUpwindInterface(halo_, faceCellsD_, phiF, gUx[k], gUy[k], gUz[k],
                                     dOwnD_, dNeiD_, corr);                     // + cut faces
            if (lust_)   // 0.25*linearUpwind + 0.75*linear, cut faces included on BOTH parts
            {
                deviceScale(corr, 0.25);
                DeviceBuffer<scalar> lc;
                deviceLinearCorr(dm_, phiInt_, Uk_[k], lc);                     // internal faces
                deviceLinearCorrInterface(halo_, faceCellsD_, phiF, weightsD_, Uk_[k], lc);   // + cut faces
                deviceAxpy(0.75, lc, corr);
            }
            if (ami_.active) deviceAmiAddLinUpwindCorr(ami_.local, k, gUxE, gUyE, gUzE, corr);   // + AMI-face linearUpwind
            dumpStage("corr_final", corr, k);
            deviceAxpy(-1.0, corr, relaxSrc[k]);                                // source -= corr
        }
    }

    // Non-orthogonal ("corrected") laplacian: the matrix above used nonOrthDeltaCoeffs (implicit); this is
    // the matching EXPLICIT deferred correction, internal faces + cut faces. It must land in relaxSrc -- the
    // same source H() reads -- like every other explicit momentum term. Mirrors the single-GPU
    // `if (ctl_.nonOrth) { deviceLaplacianCorr; +cyclic; relaxSrc -= corr; }`.
    if (nonOrth_)
    {
        DeviceBuffer<scalar> ggx[3], ggy[3], ggz[3];
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> ub;
            deviceBCValue(dbU_.comp[k], Uk_[k], ub);
            halo_.exchange(Uk_[k].data());
            halo_.scatterBoundaryValues(Uk_[k].data(), weightsD_, procStart_, ub.data());
            halo_.waitExchange();
            deviceGaussGrad(dm_, Uk_[k], ub, ggx[k], ggy[k], ggz[k]);
        }
        // AMI non-orth laplacian correction reads the rotated-TENSOR neighbour stencil of all 3 components' gradients
        // AND interp(nuEff) at the neighbour -> gather the 9 gradients + nuEff (all extended).
        DeviceBuffer<scalar> ggxE[3], ggyE[3], ggzE[3], nuE;
        if (ami_.active)
        {
            for (int l = 0; l < 3; ++l)
            {
                distributedAmiGather(ami_, ggx[l], ggxE[l]);
                distributedAmiGather(ami_, ggy[l], ggyE[l]);
                distributedAmiGather(ami_, ggz[l], ggzE[l]);
            }
            distributedAmiGather(ami_, nuEffCell, nuE);
        }
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> lc;
            deviceLaplacianCorr(dm_, nuEff_f, ggx[k], ggy[k], ggz[k], lc);              // internal faces
            deviceLapCorrInterface(halo_, faceCellsD_, nuFaceD_, magSfD_, corrVecD_,
                                   weightsD_, ggx[k], ggy[k], ggz[k], lc);              // + cut faces
            if (ami_.active) deviceAmiAddLapCorr(ami_.local, k, nuE, ggxE, ggyE, ggzE, lc);   // + AMI-face non-orth
            deviceAxpy(-1.0, lc, relaxSrc[k]);   // momentum is div - laplacian: source -= lapCorr
        }
    }

    // ---- grad(p): coupled processor boundary value from the halo ----
    DeviceBuffer<scalar> pbv;
    deviceBCValue(dbP_, dp_, pbv);
    halo_.exchange(dp_.data());
    halo_.scatterBoundaryValues(dp_.data(), weightsD_, procStart_, pbv.data());
    halo_.waitExchange();
    DeviceBuffer<scalar> gx, gy, gz;
    deviceGaussGrad(dm_, dp_, pbv, gx, gy, gz);
    if (ami_.active)   // AMI-face contribution to grad(p): grad[own] += Sf*(w*p[own]+(1-w)*interp(p))/V
    {
        DeviceBuffer<scalar> pe;
        distributedAmiGather(ami_, dp_, pe);
        deviceAmiAddGrad(ami_.local, pe, dm_.V, gx, gy, gz);
    }
    DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };

    // ---- momentum predictor ----
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> s;
        deviceHadamard(s, dm_.V, *gg[k]);
        deviceScale(s, -1.0);
        deviceAxpy(1.0, relaxSrc[k], s);
        // deferred off-diagonal rotation mixing (pairs with the diagonal ifCoeffC): PREDICTOR RHS only -- into `s`,
        // NOT relaxSrc, so UEqn.H() below stays the exact full-rotation coupling (deviceAmiAddH on the post-solve U).
        if (ami_.active && ami_.local.rotational)
            deviceAmiAddDeferredRot(ami_.local, amiUint[0], amiUint[1], amiUint[2], k, s);
        DeviceBuffer<scalar> diagC, b;
        deviceFold(dm_, mDiagR, s, iC[k], bC[k], diagC, b);
        const DeviceLduView mv = deviceLduView(dm_, diagC, mUp, mLo);
        const DistributedAMI* amip = ami_.active ? &ami_ : nullptr;
        if (ami_.active) ami_.amulComp = k;   // ROTATIONAL momentum matvec uses ifCoeffC[k] (pairs with the deferred rot)
        const scalar nf = deviceParallelNormFactor(mv, halo_, ifCoeff, Uk_[k], b, ones_, P_.globalNCells, amip);
        if (!bicgGCache_[k]) bicgGCache_[k] = std::make_unique<BiCGGraphCache>();   // lazy: whole-loop graph (BRAE_PARALLEL_GRAPH=3)
        const DeviceSolverPerf up =
            deviceParallelJacobiBiCGStab(mv, halo_, ifCoeff, b, Uk_[k], nf, tolU_, relTolU_, maxIter_, 1, bicgGCache_[k].get(), amip);
        kIters_ += up.nIterations;
        if (validC_[k] && up.initialResidual > res.Ux) res.Ux = up.initialResidual;
        dumpStage("Upred", Uk_[k], k);
    }

    // ---- rAU, HbyA ----
    DeviceBuffer<scalar> cmptAvIC;
    deviceCopy(cmptAvIC, iC[0]);
    deviceAxpy(1.0, iC[1], cmptAvIC);
    deviceAxpy(1.0, iC[2], cmptAvIC);
    deviceScale(cmptAvIC, 1.0 / 3.0);
    DeviceBuffer<scalar> diagA, dumb, rAU;
    deviceFold(dm_, mDiagR, zeroSrc_, cmptAvIC, zeroBnd_, diagA, dumb);
    deviceReciprocalV(dm_, diagA, rAU);
    dumpStage("rAU", rAU);

    DeviceBuffer<scalar> HbyA[3];
    const DeviceLduView av = deviceLduView(dm_, diagA, mUp, mLo);
    // SIMPLEC (consistent_): rAtU = 1/(1/rAU - H1) = V/max(A*1, 0.1*diagA), where A*1 is the row sum INCLUDING the
    // interface off-diagonals (deviceParallelAmul with ones). drAtU = rAtU - rAU drives the flux/HbyA corrections
    // below, and rAtU replaces rAU in the pEqn -- so a `consistent yes` case (pitzDaily) is stable at its native
    // relaxP instead of diverging as plain SIMPLE. For plain SIMPLE (consistent_ off), rAtU == rAU (a no-op copy).
    DeviceBuffer<scalar> rAtU, drAtU;
    if (consistent_)
    {
        DeviceBuffer<scalar> rowSum;
        deviceParallelAmul(av, halo_, ifCoeff, ones_, rowSum);
        deviceSimplecRAtU(dm_, rowSum, diagA, rAtU);
        deviceCopy(drAtU, rAtU);
        deviceAxpy(-1.0, rAU, drAtU);
    }
    else
        deviceCopy(rAtU, rAU);
    // AMI-face H contribution (all AMI, not just rotational): H[own] -= ifCoeff*UN[i]/V, UN = the rotated
    // AMI-interp of the POST-SOLVE U (deviceAmiInterpolateVec handles rotation via forwardT; identity when
    // translational). Needs the gathered extended field per component.
    DeviceBuffer<scalar> amiUN[3];
    if (ami_.active)
    {
        DeviceBuffer<scalar> ux, uy, uz;
        distributedAmiGather(ami_, Uk_[0], ux);
        distributedAmiGather(ami_, Uk_[1], uy);
        distributedAmiGather(ami_, Uk_[2], uz);
        deviceAmiInterpolateVec(ami_.local, ux, uy, uz, amiUN[0], amiUN[1], amiUN[2]);
    }
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> bdDiag;
        deviceCopy(bdDiag, cmptAvIC);
        deviceAxpy(-1.0, iC[k], bdDiag);
        DeviceBuffer<scalar> Hk;
        deviceParallelMatrixH(av, dm_, halo_, faceCellsD_, ifCoeff, Uk_[k], relaxSrc[k], bdDiag, bC[k], Hk);
        if (ami_.active) deviceAmiAddH(ami_.local, amiUN[k], dm_.V, Hk);
        deviceHadamard(HbyA[k], rAU, Hk);
        dumpStage("HbyA", HbyA[k], k);
    }

    // ---- phiHbyA ----
    DeviceBuffer<scalar> hb[3];
    for (int k = 0; k < 3; ++k)
    {
        deviceBCValue(dbU_.comp[k], HbyA[k], hb[k]);
        halo_.exchange(HbyA[k].data());
        halo_.scatterBoundaryValues(HbyA[k].data(), weightsD_, procStart_, hb[k].data());
        halo_.waitExchange();
    }
    DeviceBuffer<scalar> phiHbyAint, phiHbyAbnd;
    deviceVectorFlux(dm_, HbyA[0], HbyA[1], HbyA[2], phiHbyAint);
    deviceBoundaryFlux(dm_, hb[0], hb[1], hb[2], phiHbyAbnd);
    if (ami_.active)   // AMI-face phiHbyA = (w*HbyA[own] + (1-w)*interp(forwardT.HbyA[nbr])).Sf -> ami_.phi
    {
        DeviceBuffer<scalar> hxe, hye, hze;
        distributedAmiGather(ami_, HbyA[0], hxe);
        distributedAmiGather(ami_, HbyA[1], hye);
        distributedAmiGather(ami_, HbyA[2], hze);
        deviceAmiFlux(ami_.local, hxe, hye, hze);
    }

    // SIMPLEC flux + HbyA consistency correction (OF simpleFoam `consistent` branch), from the ENTRY grad(p):
    //   phiHbyA += interp(rAtU-rAU)*snGrad(p)*magSf  (= the drAtU-weighted Laplacian flux of dp_, internal + boundary)
    //   HbyA    -= (rAU-rAtU)*grad(p)  ==  HbyA += drAtU*grad(p)   (feeds the velocity corrector only)
    // phiHbyA above was built from the UNCORRECTED HbyA, so the corrections are added separately. No-op when off.
    if (consistent_)
    {
        DeviceBuffer<scalar> drAtUf;
        deviceInterpolate(dm_, drAtU, drAtUf);
        DeviceBuffer<scalar> ld, lu, ll, fInt;
        deviceLaplacianCoeffs(dm_, drAtUf, ld, lu, ll, nonOrth_);
        deviceMatrixFluxInternal(deviceLduView(dm_, ld, lu, ll), dp_, fInt);
        deviceAxpy(1.0, fInt, phiHbyAint);
        if (nonOrth_)   // explicit non-orth part of the SIMPLEC flux correction: interp(drAtU)*(corrVec.grad(p))*magSf
        {
            DeviceBuffer<scalar> ffcS;
            deviceLaplacianCorrFlux(dm_, drAtUf, gx, gy, gz, ffcS);
            deviceAxpy(1.0, ffcS, phiHbyAint);
        }
        DeviceBuffer<scalar> dIC, dBC, fBnd;
        deviceBCLaplacianCoeffs(dbP_, drAtU, dIC, dBC);
        deviceMatrixFluxBoundary(dbP_, dIC, dBC, dp_, fBnd);
        deviceAxpy(1.0, fBnd, phiHbyAbnd);
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> t;
            deviceHadamard(t, drAtU, *gg[k]);
            deviceAxpy(1.0, t, HbyA[k]);
        }
    }

    // MRF.makeRelative(phiHbyA): subtract the static frame flux on the zone faces so the pressure equation is
    // solved for the RELATIVE flux inside the rotating zone (the corrector below reconstructs the relative phi,
    // and setMRF already made the resident phi relative). No-op outside the zone.
    if (mrf_.active) deviceMrfApplyFrameFlux(mrf_, +1.0, phiHbyAint, phiHbyAbnd);

    // ---- pEqn: laplacian(rAtU,p) == div(phiHbyA)  (rAtU == rAU for plain SIMPLE) ----
    DeviceBuffer<scalar> rAUf_int;
    deviceInterpolate(dm_, rAtU, rAUf_int);
    DeviceBuffer<scalar> rAUbnd;
    deviceBCValue(dbRAU_, rAtU, rAUbnd);
    halo_.exchange(rAtU.data());
    halo_.scatterBoundaryValues(rAtU.data(), weightsD_, procStart_, rAUbnd.data());
    halo_.waitExchange();

    DeviceBuffer<scalar> pD;
    DeviceBuffer<scalar>& pU_ = pUp_;   // persistent (graph-referenced): stable address -> whole-loop graph captures once
    DeviceBuffer<scalar>& pL_ = pLp_;
    deviceLaplacianCoeffs(dm_, rAUf_int, pD, pU_, pL_, nonOrth_);
    // pEqn interface coeff ON-DEVICE (Phase A): pCoeffGeo[i] = rAU-at-cut * geomIf_[i], the same fused device
    // multiply as the momentum coeff above -- replaces the per-step rAUbnd.host() round-trip + host loop.
    std::vector<DeviceBuffer<scalar>> pCoeffGeo(nIface);
    {
        constexpr int TPB = 128;
        for (int i = 0; i < nIface; ++i)
        {
            const int n = static_cast<int>(halo_.size(i));
            if (n <= 0) continue;
            pCoeffGeo[i].resize(n);
            detail::interfaceCoeffKernel<<<(n + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
                pCoeffGeo[i].data(), rAUbnd.data(), static_cast<int>(procStart_[i]), geomIf_[i].data(), n);
        }
    }
    std::vector<DeviceBuffer<scalar>>& pIfCoeff = pIfCoeffp_;   // persistent (graph-referenced)
    deviceLaplacianInterface(halo_, faceCellsD_, pCoeffGeo, pD, pIfCoeff);

    DeviceBuffer<scalar> dphi, psrc;
    deviceDiv(dm_, phiHbyAint, phiHbyAbnd, dphi);
    if (ami_.active) deviceAmiAddDiv(ami_.local, dm_.V, dphi);   // AMI-face phiHbyA into continuity div
    deviceHadamard(psrc, dm_.V, dphi);
    DeviceBuffer<scalar> piC, pbC;
    deviceBCLaplacianCoeffsFace(dbP_, rAUbnd, piC, pbC);
    DeviceBuffer<scalar>& pDiagC = pDiagp_;   // persistent (graph-referenced)
    DeviceBuffer<scalar>& pb     = pbp_;
    deviceFold(dm_, pD, psrc, piC, pbC, pDiagC, pb);
    if (ami_.active)   // AMI pressure laplacian: ifCoeff = rAtUFace*dc*magSf; pDiagC[own] -= ifCoeff
    {
        DeviceBuffer<scalar> re;
        distributedAmiGather(ami_, rAtU, re);
        deviceAmiAssembleLaplacian(ami_.local, re, pDiagC, /*addToDiag*/ true);
    }
    // Singular all-Neumann pressure (needRef_, no fixedValue-p) WITH an AMI interface: the interface-coupled system
    // is exactly singular, so the distributed CG returns garbage without regularisation. Match the single-GPU
    // (device_simple_foam: cyclic/AMI branch) with a tiny SYMMETRIC diagonal shift eps = -1e-10*mean|diag| on every
    // cell -- it lifts the null space while preserving symmetry + the zero-mean RHS; the level is pinned post-hoc.
    // mean|diag| is a GLOBAL reduction (sum|diag| over all ranks / globalNCells) so eps is identical on every rank.
    if (ami_.active && needRef_)
    {
        const scalar localSum  = deviceSumMag(pDiagC);
        const scalar globalSum = Pstream::allReduce(localSum, ReduceOp::Sum);
        const scalar eps = -1e-10 * (globalSum / static_cast<scalar>(P_.globalNCells));
        deviceAxpy(eps, ones_, pDiagC);
    }

    // Explicit non-orth correction of laplacian(rAU,p), from the ENTRY grad(p) (gx/gy/gz), matching the
    // single-GPU pass-0 convention. It needs BOTH halves: b += -V*div(ffc) here, and phi -= ffc at the
    // corrector below -- the source alone leaves the reconstructed flux non-conservative on cut faces.
    DeviceBuffer<scalar> ffcP, ffcPami;   // ffcPami: AMI non-orth flux correction, applied to ami_.phi at the corrector
    std::vector<DeviceBuffer<scalar>> ffcPif;
    if (nonOrth_)
    {
        deviceLaplacianCorrFlux(dm_, rAUf_int, gx, gy, gz, ffcP);      // internal faces
        DeviceBuffer<scalar> sc;
        deviceFaceDivSource(dm_, ffcP, sc);
        deviceAxpy(1.0, sc, pb);
        if (ami_.active)   // AMI pressure non-orth: -ffc into pb[own], ffcPami for the corrector (ami_.phi -= ffcPami)
        {
            DeviceBuffer<scalar> re, gxe, gye, gze;
            distributedAmiGather(ami_, rAtU, re);
            distributedAmiGather(ami_, gx, gxe);
            distributedAmiGather(ami_, gy, gye);
            distributedAmiGather(ami_, gz, gze);
            deviceAmiLapCorrP(ami_.local, re, gxe, gye, gze, pb, ffcPami);
        }
        // rAU at the cut faces = the halo-interpolated boundary value, sliced per interface
        std::vector<DeviceBuffer<scalar>> rAUfaceD(procStart_.size());
        for (std::size_t i = 0; i < procStart_.size(); ++i)
        {
            const int n = static_cast<int>(halo_.size(static_cast<int>(i)));
            if (n <= 0) continue;
            rAUfaceD[i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(rAUfaceD[i].data(), rAUbnd.data() + procStart_[i], n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread),
                "rAU face slice");
        }
        deviceLapCorrPInterface(halo_, faceCellsD_, rAUfaceD, magSfD_, corrVecD_, weightsD_,
                                gx, gy, gz, pb, ffcPif);               // + cut faces
    }

    DeviceBuffer<scalar>& pSol = pSolp_;   // persistent: the whole-loop graph keys on pSol.data() -> capture once
    deviceCopy(pSol, dp_);
    const DeviceLduView pv = deviceLduView(dm_, pDiagC, pU_, pL_);
    if (ami_.active) ami_.amulComp = -1;   // pressure matvec is scalar -> the base ifCoeff (from assembleLaplacian)
    const DistributedAMI* pamip = ami_.active ? &ami_ : nullptr;
    const scalar nfp = deviceParallelNormFactor(pv, halo_, pIfCoeff, pSol, pb, ones_, P_.globalNCells, pamip);
    // pressure solve: local-AMG-preconditioned distributed CG (the strong preconditioner that converges on graded
    // meshes), else point-Jacobi CG. amgGalerkin re-coarsens the hierarchy from THIS step's pressure matrix
    // (pDiagC includes the interface + boundary diagonal terms; pU_/pL_ are the internal Laplacian off-diagonals).
    // Stage-1 OVERLAP: assemble the extended (owned + ghost ring) matrix from THIS step's pressure matrix, so the
    // RAS preconditioner solves on the overlapped domain. Owned block = pDiagC / pU_ / pL_; ghost diagonal = the
    // neighbour's pDiagC (halo exchange); interface off-diagonals = -pIfCoeff (a_og). Then Galerkin-recoarsen pAMGext_.
    OverlapCtx  ovc;
    OverlapCtx* ovp = nullptr;
    if (useOverlap_ && useAMG_ && nGhost_ > 0 && !ami_.active)
    {
        const int nOw = static_cast<int>(lnC_);
        cudaMemcpyAsync(exLdu_.diag.data(), pDiagC.data(), static_cast<std::size_t>(nOw) * sizeof(scalar),
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        halo_.exchange(pDiagC.data());                                     // recvData(i) = neighbour diagonal at interface
        for (int i = 0; i < halo_.nInterfaces(); ++i)
        {
            const int n = static_cast<int>(halo_.size(i));
            if (n <= 0) continue;
            cudaMemcpyAsync(exLdu_.diag.data() + nOw + ghOff_[i], halo_.recvData(i),
                            static_cast<std::size_t>(n) * sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        }
        cudaMemcpyAsync(exLdu_.upper.data(), pU_.data(), static_cast<std::size_t>(nIf_) * sizeof(scalar),
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(exLdu_.lower.data(), pL_.data(), static_cast<std::size_t>(nIf_) * sizeof(scalar),
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        constexpr int TPB = 128;
        const bool noCouple = std::getenv("BRAE_OVERLAP_NOCOUPLE") != nullptr;   // debug: 0 interface off-diag -> RAS==mode0
        if (!noCouple)
        for (int i = 0; i < static_cast<int>(pIfCoeff.size()); ++i)
        {
            const int n = static_cast<int>(halo_.size(i));
            if (n <= 0) continue;
            detail::negCopyK<<<(n + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
                exLdu_.upper.data() + nIf_ + ghOff_[i], pIfCoeff[i].data(), n);
            detail::negCopyK<<<(n + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
                exLdu_.lower.data() + nIf_ + ghOff_[i], pIfCoeff[i].data(), n);
        }
        amgGalerkin(pAMGext_, exLdu_.diag, exLdu_.upper, exLdu_.lower);
        amgPrepareFP32(pAMGext_, exLdu_.view());   // cast the extended hierarchy to FP32 mirrors (else the V-cycle runs on garbage)
        ovc.amgExt = &pAMGext_;
        ovc.exView = exLdu_.view();
        ovc.exRes  = &exRes_;
        ovc.exCorr = &exCorr_;
        ovc.nOwned = nOw;
        ovc.nGhost = nGhost_;
        ovc.ghOff.assign(ghOff_.begin(), ghOff_.end());
        ovp = &ovc;
    }
    DeviceSolverPerf pp;
    if (useAMG_)
    {
        amgGalerkin(pAMG_, pDiagC, pU_, pL_);
        pp = deviceParallelAMGPCG(pv, halo_, pIfCoeff, pb, pSol, pAMG_, nfp, tolP_, relTolP_, maxIter_, ovp);
    }
    else
        pp = deviceParallelJacobiPCG(pv, halo_, pIfCoeff, pb, pSol, nfp, tolP_, relTolP_, maxIter_, pamip);
    kIters_ += pp.nIterations;
    res.p = pp.initialResidual;
    dumpStage("pSol", pSol);

    // ---- corrector: conservative phi, relax p, U = HbyA - rAU*grad(p) ----
    DeviceBuffer<scalar> pfluxInt, pfluxBnd;
    deviceMatrixFluxInternal(pv, pSol, pfluxInt);
    deviceMatrixFluxBoundary(dbP_, piC, pbC, pSol, pfluxBnd);
    deviceParallelMatrixFluxInterface(halo_, faceCellsD_, pIfCoeff, procStart_, pSol, pfluxBnd);
    deviceAxpy(-1.0, pfluxInt, phiHbyAint);          // phi = phiHbyA - pflux
    deviceAxpy(-1.0, pfluxBnd, phiHbyAbnd);
    if (nonOrth_)   // ... and the non-orth face-flux correction, so div(phi)=0 holds on non-orthogonal faces
    {
        deviceAxpy(-1.0, ffcP, phiHbyAint);
        constexpr int TPB = 128;
        for (std::size_t i = 0; i < ffcPif.size(); ++i)
        {
            const int n = static_cast<int>(halo_.size(static_cast<int>(i)));
            if (n <= 0) continue;
            detail::subtractSliceKernel<<<(n + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
                phiHbyAbnd.data(), static_cast<int>(procStart_[i]), ffcPif[i].data(), n);
        }
    }
    if (ami_.active)   // AMI conservative flux: ami_.phi -= ifCoeff*(interp(p) - p[own]) from the SOLVED pressure
    {
        DeviceBuffer<scalar> pe;
        distributedAmiGather(ami_, pSol, pe);
        deviceAmiCorrectFlux(ami_.local, pe);
        if (nonOrth_ && ffcPami.size())
            deviceAxpy(-1.0, ffcPami, ami_.local.phi);   // AMI non-orth flux correction (matches phiInt_ -= ffcP)
    }
    deviceCopy(phiInt_, phiHbyAint);                 // maintained across iterations
    deviceCopy(phiBnd_, phiHbyAbnd);

    DeviceBuffer<scalar> pRelax;                     // p = pPrev + relaxP*(pNew - pPrev)
    deviceCopy(pRelax, pSol);
    deviceAxpy(-1.0, dp_, pRelax);
    deviceScale(pRelax, relaxP_);
    deviceAxpy(1.0, dp_, pRelax);
    deviceCopy(dp_, pRelax);

    DeviceBuffer<scalar> pbv2;
    deviceBCValue(dbP_, dp_, pbv2);
    halo_.exchange(dp_.data());
    halo_.scatterBoundaryValues(dp_.data(), weightsD_, procStart_, pbv2.data());
    halo_.waitExchange();
    DeviceBuffer<scalar> gxn, gyn, gzn;
    deviceGaussGrad(dm_, dp_, pbv2, gxn, gyn, gzn);
    if (ami_.active)   // AMI-face contribution to the corrector grad(p) (velocity update U = HbyA - rAU*grad(p))
    {
        DeviceBuffer<scalar> pe;
        distributedAmiGather(ami_, dp_, pe);
        deviceAmiAddGrad(ami_.local, pe, dm_.V, gxn, gyn, gzn);
    }
    DeviceBuffer<scalar>* gn[3] = { &gxn, &gyn, &gzn };
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> Un;
        deviceCorrector(HbyA[k], rAtU, *gn[k], Un);   // SIMPLEC: rAtU (== rAU for plain SIMPLE)
        deviceCopy(Uk_[k], Un);
        dumpStage("Ucorr", Uk_[k], k);
    }

    // ---- turbulence: update k/eps/nut on the corrected U/phi (OF runs turbulence->correct() at the end of
    // the SIMPLE iteration, so nut for the NEXT momentum comes from THIS step's fields) ----
    if (turbulent_)
    {
        // rebuild phiF from the CORRECTED flux: the phiF above was built from the pre-corrector phi (for the
        // momentum interface), but OF runs turbulence->correct() with the NEW phi. At np=1 this is moot (no
        // interface), but at np>1 the cut would otherwise convect k/eps with a STALE flux while the interior
        // used the fresh phiInt_ -- inconsistent across the cut. Rebuild it from the maintained phiBnd_.
        std::vector<DeviceBuffer<scalar>> phiFc(nIface);   // ON-DEVICE cut-flux slices (Phase A: no phiBnd_.host())
        for (int i = 0; i < nIface; ++i)
        {
            const int n = static_cast<int>(halo_.size(i));
            if (n <= 0) continue;
            phiFc[i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(phiFc[i].data(), phiBnd_.data() + procStart_[i], n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread),
                "phiFc cut slice");
        }
        if (turbModel_ == 3)   // Spalart-Allmaras (single nuTilda field; no k/eps)
        {
            parallelDeviceSpalartAllmarasCorrect(dm_, halo_, dbU_, dbNuTilda_, Uk_[0], Uk_[1], Uk_[2],
                nuTilda_, nut_, yCell_, phiInt_, phiBnd_, phiFc, faceCellsD_, procStart_, weightsD_, geomD_,
                nu_, relaxK_, tolU_, maxIter_, P_.globalNCells, ones_, bounded_, saCoeffs_, relTolKE_);
        }
        else if (turbModel_ == 2)   // k-omega SST + Langtry-Menter transition (eps_ holds omega)
        {
            parallelDeviceKOmegaSSTLMCorrect(dm_, halo_, wall_, dbU_, dbK_, dbEps_, dbReThetat_, dbGammaInt_,
                Uk_[0], Uk_[1], Uk_[2], k_, eps_, nut_, ReThetat_, gammaInt_, yCell_,
                phiInt_, phiBnd_, phiFc, faceCellsD_, procStart_, weightsD_, geomD_,
                isWallCell_, nu_, relaxEps_, relaxK_, tolU_, maxIter_, P_.globalNCells, ones_, bounded_, sstCoeffs_, relTolKE_);
        }
        else if (turbModel_ == 1)   // k-omega SST (eps_ holds omega)
        {
            parallelDeviceKOmegaSSTCorrect(dm_, halo_, wall_, dbU_, dbK_, dbEps_, Uk_[0], Uk_[1], Uk_[2],
                k_, eps_, nut_, yCell_, phiInt_, phiBnd_, phiFc, faceCellsD_, procStart_, weightsD_, geomD_,
                isWallCell_, nu_, relaxEps_, relaxK_, tolU_, maxIter_, P_.globalNCells, ones_, bounded_, sstCoeffs_,
                /*lm*/ false, /*gammaIntEff*/ nullptr, relTolKE_);
        }
        else                   // k-epsilon
        {
            scalar rEps = 0, rK = 0;
            parallelDeviceKEpsilonCorrect(dm_, halo_, wall_, dbU_, dbK_, dbEps_, Uk_[0], Uk_[1], Uk_[2],
                k_, eps_, nut_, phiInt_, phiBnd_, phiFc, faceCellsD_, procStart_, weightsD_, geomD_, isWallCell_,
                nu_, relaxEps_, relaxK_, tolU_, maxIter_, P_.globalNCells, ones_, bounded_, keCoeffs_, &rEps, &rK, relTolKE_);
        }
    }
    return res;
}

} // namespace brae
