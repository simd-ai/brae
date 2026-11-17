// Interface normal and curvature on the device -- see the header for what is here and what is not.
#include "device_interface_properties.cuh"
#include "device_processor_ops.cuh"
#include "device_glibc_math.cuh"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

inline void ckIP(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae deviceInterfaceProperties: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// interpolate -> normalise -> dot, in one pass. The face gradient is a register triple and never
// reaches memory.
__global__ void nHatfInternalKernel(
    const label*  __restrict__ own,
    const label*  __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    int nIf, scalar deltaN,
    scalar* __restrict__ nHatf)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;

    const label o = own[f], n = nei[f];
    const scalar wf = w[f], wn = scalar(1) - wf;
    const scalar fx = wf*gx[o] + wn*gx[n];
    const scalar fy = wf*gy[o] + wn*gy[n];
    const scalar fz = wf*gz[o] + wn*gz[n];

    // nHatfv = gradAlphaf/(mag + deltaN). deltaN is what keeps this finite where gradAlpha vanishes,
    // which is most of the domain -- so the divide is unguarded ON PURPOSE and the stabiliser is the
    // guard. A branch on mag == 0 here would be a second, different guard.
    const scalar m = sqrt(fx*fx + fy*fy + fz*fz);
    const scalar s = scalar(1) / (m + deltaN);

    nHatf[f] = (fx*s)*Sfx[f] + (fy*s)*Sfy[f] + (fz*s)*Sfz[f];
}

// ...and the faces of a PERIODIC PAIR, which are in neither list. OpenFOAM's line is the same one:
// gradAlphaf = fvc::interpolate(grad(alpha)), and on a coupled patch that interpolation is the two
// CELLS' gradients weighted (surfaceInterpolationScheme.C:284-293); then nHatfv = gradAlphaf/(mag +
// deltaN) and nHatf = nHatfv & Sf (interfaceProperties.C:134-150). correctContactAngle does not reach a
// cyclic: it walks the alphaContactAngle patches, and a periodic pair is not one.
__global__ void nHatfCyclicKernel(
    const label*  __restrict__ own,
    CyclicNbr nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    int n, scalar deltaN,
    scalar* __restrict__ nHatf)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j];
    const scalar wf = w[j], wn = scalar(1) - wf;
    const scalar fx = wf*gx[o] + wn*cyclicNbrValue(nbr, gx, j);
    const scalar fy = wf*gy[o] + wn*cyclicNbrValue(nbr, gy, j);
    const scalar fz = wf*gz[o] + wn*cyclicNbrValue(nbr, gz, j);

    const scalar m = sqrt(fx*fx + fy*fy + fz*fz);
    const scalar s = scalar(1) / (m + deltaN);

    nHatf[j] = (fx*s)*Sfx[j] + (fy*s)*Sfy[j] + (fz*s)*Sfz[j];
}

// The boundary faces, from normals the caller has already corrected. Sf is indexed by the GLOBAL face
// number -- boundary faces follow the internal ones in the mesh's own ordering -- which is why this
// takes nInternalFaces as well.
__global__ void nHatfBoundaryKernel(
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ nbx,
    const scalar* __restrict__ nby,
    const scalar* __restrict__ nbz,
    int nBf, int nIf,
    scalar* __restrict__ nHatfBnd)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    const int f = nIf + b;
    nHatfBnd[b] = nbx[b]*Sfx[f] + nby[b]*Sfy[f] + nbz[b]*Sfz[f];
}

__global__ void negateKernel(scalar* __restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = -x[i];
}

// ---- the contact-angle boundary (DeviceContactAngleState) ---------------------------------------------
// Separately rounded operations throughout: the host forms these in plain doubles, with no contraction.
__device__ __forceinline__ scalar mulR(scalar a, scalar b) { return __dmul_rn(a, b); }
__device__ __forceinline__ scalar addR(scalar a, scalar b) { return __dadd_rn(a, b); }
__device__ __forceinline__ scalar subR(scalar a, scalar b) { return __dsub_rn(a, b); }
__device__ __forceinline__ scalar divR(scalar a, scalar b) { return __ddiv_rn(a, b); }
// a.x*b.x + a.y*b.y + a.z*b.z, left to right
__device__ __forceinline__ scalar dot3R(scalar ax, scalar ay, scalar az, scalar bx, scalar by, scalar bz)
{
    return addR(addR(mulR(ax, bx), mulR(ay, by)), mulR(az, bz));
}

// One face's evaluate (fv_patch_field.cuh: ZeroGradientPatchField, EmptyPatchField, and
// ConstantAlphaContactAnglePatchField's limiting body over FixedGradientPatchField::evaluate). `g` is the
// face's gradient in and out; `value` the value it held before (what `limit gradient` clamps against).
__device__ __forceinline__ scalar caEvaluateFace(label kind, label lim, scalar cell, scalar& g, scalar value,
                                                 scalar dc)
{
    if (kind != 1) return cell;
    if (lim == 1)
    {
        const scalar implied = addR(value, divR(g, dc));
        const scalar clamped = implied < scalar(0) ? scalar(0) : (implied > scalar(1) ? scalar(1) : implied);
        g = mulR(dc, subR(clamped, value));
    }
    else if (lim == 2)
    {
        g = scalar(0);
    }
    const scalar v = addR(cell, divR(g, dc));
    return lim == 3 ? (v < scalar(0) ? scalar(0) : (v > scalar(1) ? scalar(1) : v)) : v;
}

__global__ void caEvaluateKernel(
    int n, const label* __restrict__ kind, const label* __restrict__ lim, const label* __restrict__ fc,
    const scalar* __restrict__ dc, const scalar* __restrict__ cells, scalar* __restrict__ refGrad,
    scalar* __restrict__ stored, int empties)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n) return;
    if (kind[b] == 2 && !empties) return;
    scalar g = refGrad[b];
    stored[b] = caEvaluateFace(kind[b], lim[b], cells[fc[b]], g, stored[b], dc[b]);
    if (kind[b] == 1) refGrad[b] = g;
}

__global__ void caRelaxKernel(
    int n, const label* __restrict__ kind, const label* __restrict__ lim, const label* __restrict__ fc,
    const scalar* __restrict__ dc, const scalar* __restrict__ postMules, const scalar* __restrict__ relaxed,
    scalar* __restrict__ refGrad, scalar* __restrict__ stored, int emptiesAverage)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n) return;
    if (kind[b] == 2 && !emptiesAverage)
    {
        stored[b] = relaxed[fc[b]];
        return;
    }
    const scalar alpha10 = stored[b];
    scalar g = refGrad[b];
    const scalar post = caEvaluateFace(kind[b], lim[b], postMules[fc[b]], g, alpha10, dc[b]);
    if (kind[b] == 1) refGrad[b] = g;
    stored[b] = addR(mulR(scalar(0.5), post), mulR(scalar(0.5), alpha10));
}

// finishNHatBoundary's gradient: a stencil cell's internal sum plus its boundary faces (ascending boundary
// index = patch order, then face order; empty faces left out), over V.
__global__ void caCellGradientKernel(
    int nSt, const label* __restrict__ stCells, const scalar* __restrict__ acc,
    const label* __restrict__ bndCellStart, const label* __restrict__ bndPerm,
    const label* __restrict__ bndGFace, const label* __restrict__ bndIsEmpty,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ stored, const scalar* __restrict__ V,
    scalar* __restrict__ gx, scalar* __restrict__ gy, scalar* __restrict__ gz)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nSt) return;
    const label cell = stCells[k];
    scalar ax = acc[k], ay = acc[k + nSt], az = acc[k + 2*nSt];
    for (label j = bndCellStart[cell]; j < bndCellStart[cell + 1]; ++j)
    {
        const label b = bndPerm[j];
        if (bndIsEmpty[b]) continue;
        const label f = bndGFace[b];
        const scalar pv = stored[b];
        ax = addR(ax, mulR(Sfx[f], pv));
        ay = addR(ay, mulR(Sfy[f], pv));
        az = addR(az, mulR(Sfz[f], pv));
    }
    gx[cell] = divR(ax, V[cell]);
    gy[cell] = divR(ay, V[cell]);
    gz[cell] = divR(az, V[cell]);
}

// nHatBoundary (interface_properties_cpp.cu) per face, first half: the corrected face gradient, its
// magnitude, the unit normal, and a contact face's a12 = nHat & nf.
__global__ void caNormalKernel(
    int n, const label* __restrict__ kind, const label* __restrict__ fc,
    const scalar* __restrict__ nfx, const scalar* __restrict__ nfy, const scalar* __restrict__ nfz,
    const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
    const scalar* __restrict__ refGrad, scalar deltaN,
    scalar* __restrict__ hxo, scalar* __restrict__ hyo, scalar* __restrict__ hzo, scalar* __restrict__ mo,
    scalar* __restrict__ a12o)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n) return;
    if (kind[b] == 2) return;
    const label c = fc[b];
    const scalar nx = nfx[b], ny = nfy[b], nz = nfz[b];
    const scalar gcx = gx[c], gcy = gy[c], gcz = gz[c];
    const scalar snA = kind[b] == 1 ? refGrad[b] : scalar(0);
    const scalar nn = dot3R(nx, ny, nz, gcx, gcy, gcz);
    const scalar d = subR(snA, nn);
    const scalar gbx = addR(gcx, mulR(nx, d));
    const scalar gby = addR(gcy, mulR(ny, d));
    const scalar gbz = addR(gcz, mulR(nz, d));
    // faceUnitNormal
    const scalar m = __dsqrt_rn(dot3R(gbx, gby, gbz, gbx, gby, gbz));
    const scalar s = divR(scalar(1), addR(m, deltaN));
    const scalar hx = mulR(gbx, s), hy = mulR(gby, s), hz = mulR(gbz, s);
    hxo[b] = hx;
    hyo[b] = hy;
    hzo[b] = hz;
    mo[b] = m;
    a12o[b] = kind[b] == 1 ? dot3R(hx, hy, hz, nx, ny, nz) : scalar(0);
}

// ...second half: correctContactAngle with b1 = cos(theta) and b2 = cos(acos(a12) - theta) as the HOST's libm
// gives them, contactAngleGradient, the patch's evaluate after setGradient, and nHatf = nHat & Sf.
__global__ void caContactKernel(
    int n, const label* __restrict__ kind, const label* __restrict__ lim, const label* __restrict__ fc,
    const scalar* __restrict__ dc, const scalar* __restrict__ cosTheta, const scalar* __restrict__ b2s,
    const scalar* __restrict__ nfx, const scalar* __restrict__ nfy, const scalar* __restrict__ nfz,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ hxi, const scalar* __restrict__ hyi, const scalar* __restrict__ hzi,
    const scalar* __restrict__ mi, const scalar* __restrict__ a12i,
    const scalar* __restrict__ cells, scalar deltaN,
    scalar* __restrict__ refGrad, scalar* __restrict__ stored, scalar* __restrict__ nHatf)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n) return;
    if (kind[b] == 2)
    {
        nHatf[b] = scalar(0);
        return;
    }
    scalar hx = hxi[b], hy = hyi[b], hz = hzi[b];
    if (kind[b] == 1)
    {
        const scalar nx = nfx[b], ny = nfy[b], nz = nfz[b];
        const scalar a12 = a12i[b];
        const scalar b1 = cosTheta[b];
        const scalar b2 = b2s[b];
        const scalar det = subR(scalar(1), mulR(a12, a12));
        const scalar A = divR(subR(b1, mulR(a12, b2)), det);
        const scalar B = divR(subR(b2, mulR(a12, b1)), det);
        const scalar vx = addR(mulR(A, nx), mulR(B, hx));
        const scalar vy = addR(mulR(A, ny), mulR(B, hy));
        const scalar vz = addR(mulR(A, nz), mulR(B, hz));
        const scalar mv = __dsqrt_rn(dot3R(vx, vy, vz, vx, vy, vz));
        const scalar s2 = divR(scalar(1), addR(mv, deltaN));
        hx = mulR(vx, s2); hy = mulR(vy, s2); hz = mulR(vz, s2);
        scalar g = mulR(dot3R(nx, ny, nz, hx, hy, hz), mi[b]);
        stored[b] = caEvaluateFace(1, lim[b], cells[fc[b]], g, stored[b], dc[b]);
        refGrad[b] = g;
    }
    const label f = bndGFace[b];
    nHatf[b] = dot3R(hx, hy, hz, Sfx[f], Sfy[f], Sfz[f]);
}

// The faces whose a12 differs, bit for bit, from the last call's: their index and new value are listed and the
// last call's array brought up to date. b2 is a function of a12 and the face's own constants and nothing else, so
// a face whose a12 has not moved keeps its b2 exactly.
__global__ void caChangedKernel(
    int n, const scalar* __restrict__ a12, scalar* __restrict__ prev, int force,
    int* __restrict__ count, int* __restrict__ idx, scalar* __restrict__ val)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar a = a12[i];
    if (force || __double_as_longlong(a) != __double_as_longlong(prev[i]))
    {
        const int k = atomicAdd(count, 1);
        idx[k] = i;
        val[k] = a;
        prev[i] = a;
    }
}

__global__ void caScatterKernel(
    int m, const int* __restrict__ idx, const scalar* __restrict__ val, scalar* __restrict__ out)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= m) return;
    out[idx[k]] = val[k];
}

// updateMixtureBoundary (inter_case_cpp.cu), uncoupled branch, with alpha2 = 1 - alpha1
__global__ void caMixtureKernel(
    int n, const scalar* __restrict__ a, scalar rho1, scalar nu1, scalar rho2, scalar nu2,
    scalar* __restrict__ rho, scalar* __restrict__ mu, scalar* __restrict__ nu)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n) return;
    const scalar ab = a[b];
    const scalar lo = ab < scalar(0) ? scalar(0) : ab;
    const scalar ac = scalar(1) < lo ? scalar(1) : lo;
    const scalar a2 = subR(scalar(1), ab);
    rho[b] = addR(mulR(ab, rho1), mulR(a2, rho2));
    const scalar oneMinus = subR(scalar(1), ac);
    const scalar m = addR(mulR(mulR(ac, rho1), nu1), mulR(mulR(oneMinus, rho2), nu2));
    mu[b] = m;
    nu[b] = divR(m, addR(mulR(ac, rho1), mulR(oneMinus, rho2)));
}

__global__ void caDivCoeffsKernel(
    int n, const label* __restrict__ kind, const scalar* __restrict__ refGrad, const scalar* __restrict__ dc,
    const scalar* __restrict__ flux, scalar* __restrict__ iC, scalar* __restrict__ bC)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n) return;
    const scalar pf = flux[b];
    const scalar vBC = kind[b] == 1 ? divR(refGrad[b], dc[b]) : scalar(0);
    iC[b] = mulR(pf, scalar(1));
    bC[b] = mulR(-pf, vBC);
}

__global__ void caSurfaceTensionKernel(
    int n, const label* __restrict__ kind, const label* __restrict__ fc, const scalar* __restrict__ refGrad,
    const scalar* __restrict__ cells, const scalar* __restrict__ K, scalar sigma, scalar* __restrict__ stf)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n) return;
    const label c = fc[b];
    const scalar g = kind[b] == 1 ? refGrad[b] : scalar(0);
    const scalar snA = addR(mulR(scalar(0), cells[c]), g);
    stf[b] = mulR(mulR(sigma, K[c]), snA);
}

}   // namespace


void deviceInterfaceNormalFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfInt)
{
    if (nInternalFaces <= 0) return;
    nHatfInt.resize(static_cast<std::size_t>(nInternalFaces));
    nHatfInternalKernel<<<nBlocks(nInternalFaces), TPB>>>(
        dm.owner.data(), dm.nei.data(), dm.w.data(),
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        gx.data(), gy.data(), gz.data(),
        nInternalFaces, deltaN, nHatfInt.data());
    ckIP(cudaGetLastError(), "nHatf internal");
}


void deviceInterfaceNormalFluxBoundary(
    const DeviceMesh&           dm,
    int                         nBoundaryFaces,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& nbx,
    const DeviceBuffer<scalar>& nby,
    const DeviceBuffer<scalar>& nbz,
    DeviceBuffer<scalar>&       nHatfBnd)
{
    if (nBoundaryFaces <= 0) return;
    nHatfBnd.resize(static_cast<std::size_t>(nBoundaryFaces));
    nHatfBoundaryKernel<<<nBlocks(nBoundaryFaces), TPB>>>(
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        nbx.data(), nby.data(), nbz.data(),
        nBoundaryFaces, nInternalFaces, nHatfBnd.data());
    ckIP(cudaGetLastError(), "nHatf boundary");
}


void deviceInterfaceCurvature(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& nHatfInt,
    const DeviceBuffer<scalar>& nHatfBnd,
    DeviceBuffer<scalar>&       K)
{
    deviceDiv(dm, nHatfInt, nHatfBnd, K);
    const int n = static_cast<int>(K.size());
    if (n <= 0) return;
    negateKernel<<<nBlocks(n), TPB>>>(K.data(), n);
    ckIP(cudaGetLastError(), "curvature negate");
}


void deviceInterfaceNormalFluxCyclic(
    DeviceCyclic&               cyc,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfIf)
{
    if (cyc.n == 0) { nHatfIf.resize(0); return; }
    nHatfIf.resize(static_cast<std::size_t>(cyc.n));
    nHatfCyclicKernel<<<nBlocks(cyc.n), TPB>>>(
        cyc.ownCell.data(), cyc.nbr(), cyc.weights.data(),
        cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(),
        gx.data(), gy.data(), gz.data(), cyc.n, deltaN, nHatfIf.data());
    ckIP(cudaGetLastError(), "nHatf, interface");
}


void deviceInterfaceCorrect(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& alpha1,
    const DeviceBuffer<scalar>& alpha1Bnd,
    DeviceBuffer<scalar>&       nHatfBnd,
    scalar                      deltaN,
    bool                        gradLeastSquares,
    scalar                      gradCellLimitK,
    DeviceBuffer<scalar>&       nHatfInt,
    DeviceBuffer<scalar>&       K,
    DeviceCyclic*               cyc,
    DeviceBuffer<scalar>*       nHatfIf)
{
    const bool pair = cyc && cyc->n > 0;
    // a limiter that cannot see the pair's faces: see deviceAlphaCorrector's refusal of the same
    if (pair && gradCellLimitK > scalar(0))
        throw std::runtime_error(
            "brae interfaceProperties (device): fvSchemes names a cellLimited nHat gradient on a mesh "
            "with a periodic pair. The limiter would not see the pair's faces. Refused rather than "
            "limit a gradient without them.");
    DeviceProcCtx* const proc = DeviceProcCtx::active(dm) ? DeviceProcCtx::current() : nullptr;
    if (proc && (gradLeastSquares || gradCellLimitK > scalar(0)))
        throw std::runtime_error(
            "brae interfaceProperties (device): a leastSquares or cellLimited nHat gradient across processor faces. "
            "Only `Gauss linear` is ported under -parallel.");
    DeviceBuffer<scalar> gx, gy, gz;
    // fvc::grad(alpha1_, "nHat") by the case's own entry: a leastSquares fit takes the pair inside it
    deviceGradOf(dm, alpha1, alpha1Bnd, gradLeastSquares, gradCellLimitK, gx, gy, gz,
                 (gradLeastSquares && pair) ? cyc : nullptr);
    // the gradient's halo, for the neighbour's side of nHatf on the processor faces
    if (proc) procExchange(*proc, {&gx, &gy, &gz}, "grad(alpha1)");
    // THE PAIR's own contribution to a Gauss grad(alpha) first: nHatf is built from that gradient, and
    // a gradient without the pair is the gradient of a mesh with a wall there.
    if (!gradLeastSquares && pair)
    {
        deviceCyclicAddGrad(*cyc, alpha1, dm.V, gx, gy, gz);
    }
    deviceInterfaceNormalFlux(dm, dm.nInternalFaces, gx, gy, gz, deltaN, nHatfInt);
    if (cyc && cyc->n > 0)
    {
        if (!nHatfIf)
        {
            throw std::runtime_error(
                "brae interfaceProperties (device): the mesh has a periodic pair and no array was given "
                "for nHatf there. The curvature is -div(nHatf), which sums a coupled face like any "
                "other, and the compressive flux reads the same normal.");
        }
        deviceInterfaceNormalFluxCyclic(*cyc, gx, gy, gz, deltaN, *nHatfIf);
    }
    if (proc)
    {
        DeviceBuffer<scalar> nHatfProc;
        deviceInterfaceNormalFluxCyclic(proc->cyc(), gx, gy, gz, deltaN, nHatfProc);
        procScatter(*proc, nHatfProc, nHatfBnd);
    }
    deviceInterfaceCurvature(dm, nHatfInt, nHatfBnd, K);
    if (proc) procExchange(*proc, {&K}, "K");
    // ...and the pair's own face in -div(nHatf): K is a divergence, and fvc::div sums a coupled patch's
    // face into its cell like any patch's (fvc.cu:548-550). The negate above has already run, so this
    // subtracts what that sum would have added.
    if (cyc && cyc->n > 0)
    {
        DeviceBuffer<scalar> negIf(static_cast<std::size_t>(cyc->n));
        ckIP(cudaMemcpy(negIf.data(), nHatfIf->data(), sizeof(scalar)*cyc->n,
                        cudaMemcpyDeviceToDevice), "nHatf copy, interface");
        negateKernel<<<nBlocks(cyc->n), TPB>>>(negIf.data(), cyc->n);
        ckIP(cudaGetLastError(), "nHatf negate, interface");
        deviceCyclicAddDivFlux(*cyc, negIf, dm.V, K);
    }
}


void deviceContactAngleEvaluate(DeviceContactAngleState& s, const DeviceBuffer<scalar>& cells, bool empties)
{
    if (s.n <= 0) return;
    caEvaluateKernel<<<nBlocks(s.n), TPB>>>(s.n, s.kind.data(), s.limit.data(), s.faceCell.data(),
                                           s.deltaCoeffs.data(), cells.data(), s.refGrad.data(), s.stored.data(),
                                           empties ? 1 : 0);
    ckIP(cudaGetLastError(), "contact angle: evaluate");
}

void deviceContactAngleRelax(DeviceContactAngleState& s, const DeviceBuffer<scalar>& postMules,
                             const DeviceBuffer<scalar>& relaxed, bool emptiesAverage)
{
    if (s.n <= 0) return;
    caRelaxKernel<<<nBlocks(s.n), TPB>>>(s.n, s.kind.data(), s.limit.data(), s.faceCell.data(),
                                        s.deltaCoeffs.data(), postMules.data(), relaxed.data(),
                                        s.refGrad.data(), s.stored.data(), emptiesAverage ? 1 : 0);
    ckIP(cudaGetLastError(), "contact angle: relaxed assignment");
}

void deviceContactAngleNHat(const DeviceMesh& dm, DeviceContactAngleState& s,
                            const DeviceBuffer<scalar>& cells, const DeviceBuffer<label>& stCells,
                            const DeviceBuffer<scalar>& acc, int nSt, scalar deltaN,
                            DeviceBuffer<scalar>& nHatfBnd)
{
    const std::size_t nC = static_cast<std::size_t>(dm.nCells);
    s.gx.resize(nC);
    s.gy.resize(nC);
    s.gz.resize(nC);
    if (nSt > 0)
    {
        caCellGradientKernel<<<nBlocks(nSt), TPB>>>(
            nSt, stCells.data(), acc.data(), dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(),
            dm.bndIsEmpty.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), s.stored.data(), dm.V.data(),
            s.gx.data(), s.gy.data(), s.gz.data());
        ckIP(cudaGetLastError(), "contact angle: the boundary cells' gradient");
    }
    // the physical faces are the prefix [0, s.n); a rank with processor faces holds them after it, in the same array
    // (the interface kernel writes that segment: deviceInterfaceCorrect)
    nHatfBnd.resize(dm.nHalo > 0 ? static_cast<std::size_t>(dm.nBndFaces) : static_cast<std::size_t>(s.n));
    if (s.n <= 0) return;
    const std::size_t n = static_cast<std::size_t>(s.n);
    s.hx.resize(n); s.hy.resize(n); s.hz.resize(n); s.mag.resize(n); s.a12.resize(n); s.b2.resize(n);
    caNormalKernel<<<nBlocks(s.n), TPB>>>(
        s.n, s.kind.data(), s.faceCell.data(), s.nfx.data(), s.nfy.data(), s.nfz.data(),
        s.gx.data(), s.gy.data(), s.gz.data(), s.refGrad.data(), deltaN,
        s.hx.data(), s.hy.data(), s.hz.data(), s.mag.data(), s.a12.data());
    ckIP(cudaGetLastError(), "contact angle: the boundary normal");
    // b2 = cos(acos(a12) - theta) ON THE DEVICE, bit for bit what the host's glibc gives (device_glibc_math.cuh): glibc's
    // own algorithm and tables, in the variant that this host's libm turns out to be, found by a self-test at the first
    // call. No copy and no sync: the kernel is queued with the rest. Where no variant reproduces the host's libm the
    // host loop below runs, as it always did.
    //   BRAE_CONTROL_CA_B2_HOST=1 takes the host loop
    static const bool b2Host = std::getenv("BRAE_CONTROL_CA_B2_HOST") != nullptr;
    if (s.b2Variant == -2)
    {
        s.b2Variant = -1;
        if (!b2Host && s.hostTheta.size() == n && s.hostKind.size() == n)
        {
            std::vector<scalar> thetas;
            for (std::size_t i = 0; i < n && thetas.size() < 16; ++i)
            {
                if (s.hostKind[i] == 1 && std::find(thetas.begin(), thetas.end(), s.hostTheta[i]) == thetas.end())
                {
                    thetas.push_back(s.hostTheta[i]);
                }
            }
            s.theta.copyFrom(s.hostTheta);
            s.b2Variant = glibcB2Variant(thetas);
            std::printf("  contact angle: b2 = cos(acos(a12) - theta) on the GPU with %s%s\n",
                        glibcB2VariantName(s.b2Variant),
                        s.b2Variant < 0 ? " (no device variant reproduces this host's libm bit for bit)" : "");
        }
    }
    if (s.b2Variant >= 0)
    {
        glibcB2(s.b2Variant, s.n, s.kind.data(), s.theta.data(), s.a12.data(), s.b2.data());
    }
    // b2 = cos(acos(a12) - theta) by the host's libm, which is what the host hooks take. CUDA's acos and cos
    // are not glibc's: MEASURED on 2e7 arguments, acos differs in its last bit on 19.6% and cos(acos(x)) on 13%,
    // and on G4 that moved the wall gradient by an ulp on ~100 faces a pass, fed back by the next. So the libm
    // half stays on the host -- but only for the faces whose a12 moved since the last call. b2 depends on a12 and
    // the face's own constants alone, so an unmoved face keeps its b2 bit for bit; on G4 the interface touches a
    // few thousand of 533,594 wall faces. MEASURED with `nsys`: 75% of the host time in the GPU's idle gaps (27%
    // of the step) was this loop and the two whole-array copies around it.
    //   BRAE_CONTROL_CA_B2_FULL=1 takes the whole array through the host's libm at every call, as before
    if (s.b2Variant < 0)
    {
        static const bool b2Full = std::getenv("BRAE_CONTROL_CA_B2_FULL") != nullptr;
        if (!s.hostA12)
        {
            ckIP(cudaMallocHost(&s.hostA12, n*sizeof(scalar)), "pinned a12");
            ckIP(cudaMallocHost(&s.hostB2, n*sizeof(scalar)), "pinned b2");
            ckIP(cudaMallocHost(&s.hostIdx, n*sizeof(int)), "pinned changed indices");
        }
        const label* const kindh = s.hostKind.data();
        const scalar* const thetah = s.hostTheta.data();
        if (b2Full)
        {
            ckIP(cudaMemcpy(s.hostA12, s.a12.data(), n*sizeof(scalar), cudaMemcpyDeviceToHost), "a12 down");
            const scalar* const a12h = s.hostA12;
            scalar* const b2h = s.hostB2;
            const long nl = static_cast<long>(n);
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < nl; ++i)
            {
                b2h[i] = kindh[i] == 1 ? std::cos(std::acos(a12h[i]) - thetah[i]) : scalar(0);
            }
            ckIP(cudaMemcpy(s.b2.data(), b2h, n*sizeof(scalar), cudaMemcpyHostToDevice), "b2 up");
        }
        else
        {
            if (s.prevA12.size() != n)
            {
                s.prevA12.resize(n);
                s.changedVal.resize(n);
                s.changedIdx.resize(n);
                s.changedCount.resize(1);
                s.a12Valid = false;
            }
            ckIP(cudaMemsetAsync(s.changedCount.data(), 0, sizeof(int), cudaStreamPerThread), "changed count zero");
            caChangedKernel<<<nBlocks(s.n), TPB>>>(
                s.n, s.a12.data(), s.prevA12.data(), s.a12Valid ? 0 : 1, s.changedCount.data(), s.changedIdx.data(),
                s.changedVal.data());
            ckIP(cudaGetLastError(), "contact angle: the faces whose a12 moved");
            int m = 0;
            ckIP(cudaMemcpy(&m, s.changedCount.data(), sizeof(int), cudaMemcpyDeviceToHost), "changed count down");
            s.a12Valid = true;
            if (m > 0)
            {
                ckIP(cudaMemcpy(s.hostIdx, s.changedIdx.data(), static_cast<std::size_t>(m)*sizeof(int),
                                cudaMemcpyDeviceToHost), "changed indices down");
                ckIP(cudaMemcpy(s.hostA12, s.changedVal.data(), static_cast<std::size_t>(m)*sizeof(scalar),
                                cudaMemcpyDeviceToHost), "changed a12 down");
                const scalar* const a12h = s.hostA12;
                scalar* const b2h = s.hostB2;
                const int* const idxh = s.hostIdx;
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < m; ++j)
                {
                    const int i = idxh[j];
                    b2h[j] = kindh[i] == 1 ? std::cos(std::acos(a12h[j]) - thetah[i]) : scalar(0);
                }
                ckIP(cudaMemcpy(s.changedVal.data(), b2h, static_cast<std::size_t>(m)*sizeof(scalar),
                                cudaMemcpyHostToDevice), "changed b2 up");
                caScatterKernel<<<(m + TPB - 1)/TPB, TPB>>>(m, s.changedIdx.data(), s.changedVal.data(), s.b2.data());
                ckIP(cudaGetLastError(), "contact angle: the changed b2");
            }
        }
    }
    caContactKernel<<<nBlocks(s.n), TPB>>>(
        s.n, s.kind.data(), s.limit.data(), s.faceCell.data(), s.deltaCoeffs.data(), s.cosTheta.data(),
        s.b2.data(), s.nfx.data(), s.nfy.data(), s.nfz.data(), dm.bndGFace.data(), dm.Sfx.data(), dm.Sfy.data(),
        dm.Sfz.data(), s.hx.data(), s.hy.data(), s.hz.data(), s.mag.data(), s.a12.data(), cells.data(), deltaN,
        s.refGrad.data(), s.stored.data(), nHatfBnd.data());
    ckIP(cudaGetLastError(), "contact angle: the contact angle and nHatf");
}

void deviceContactAngleMixture(const DeviceBuffer<scalar>& aBnd, scalar rho1, scalar nu1, scalar rho2,
                               scalar nu2, DeviceBuffer<scalar>& rhoBnd, DeviceBuffer<scalar>& muBnd,
                               DeviceBuffer<scalar>& nuBnd)
{
    const int n = static_cast<int>(aBnd.size());
    rhoBnd.resize(aBnd.size());
    muBnd.resize(aBnd.size());
    nuBnd.resize(aBnd.size());
    if (n <= 0) return;
    caMixtureKernel<<<nBlocks(n), TPB>>>(n, aBnd.data(), rho1, nu1, rho2, nu2, rhoBnd.data(), muBnd.data(),
                                        nuBnd.data());
    ckIP(cudaGetLastError(), "contact angle: the boundary mixture");
}

void deviceContactAngleDivCoeffs(const DeviceContactAngleState& s, const DeviceBuffer<scalar>& flux,
                                 DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
{
    iC.resize(static_cast<std::size_t>(s.n));
    bC.resize(static_cast<std::size_t>(s.n));
    if (s.n <= 0) return;
    caDivCoeffsKernel<<<nBlocks(s.n), TPB>>>(s.n, s.kind.data(), s.refGrad.data(), s.deltaCoeffs.data(),
                                            flux.data(), iC.data(), bC.data());
    ckIP(cudaGetLastError(), "contact angle: divergence coefficients");
}

void deviceContactAngleSurfaceTension(const DeviceContactAngleState& s, const DeviceBuffer<scalar>& cells,
                                      const DeviceBuffer<scalar>& K, scalar sigma, scalar* stfBnd)
{
    if (s.n <= 0) return;
    caSurfaceTensionKernel<<<nBlocks(s.n), TPB>>>(s.n, s.kind.data(), s.faceCell.data(), s.refGrad.data(),
                                                 cells.data(), K.data(), sigma, stfBnd);
    ckIP(cudaGetLastError(), "contact angle: the surface-tension flux on the boundary");
}
} // namespace brae
