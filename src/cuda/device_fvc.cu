// cf GPU offload (G3): fvc explicit operators on device. interpolate is per-internal-face; div and
// gaussGrad are per-cell gathers (internal owner/neighbour faces via ownerStart/losort, boundary faces via
// bndCellStart), race-free, deterministic, matching the CPU fvc to machine precision.
#include "device_mesh.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
void interpKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ vol,
    scalar* __restrict__ sf)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf) sf[f] = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
}


__global__
void divKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ phiInt,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,   // emptyFvPatch::size() == 0: never in OpenFOAM's surfaceIntegrate
    const scalar* __restrict__ bval,
    const scalar* __restrict__ V,
    scalar* __restrict__ d)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // A TRANSCRIPTION of the host's fvc::div (fvc.cu): `d[own] += phi; d[nei] -= phi` in FACE ORDER from
    // zero, so this cell's owner and neighbour lists (each ascending) are merged by face index, then the
    // boundary faces in patch order, then the division by V. Summed owner-first it differed from the
    // host's in the last bit (tests/test_device_laplacian_vs_host.cu holds the two to memcmp).
    scalar s = 0.0;
    int fo = ownerStart[c];
    const int foEnd = ownerStart[c + 1];
    int kn = losortStart[c];
    const int knEnd = losortStart[c + 1];
    while (fo < foEnd || kn < knEnd)
    {
        if ((kn >= knEnd) || (fo < foEnd && fo < losort[kn]))
        {
            s += phiInt[fo];            // +owner internal
            ++fo;
        }
        else
        {
            s -= phiInt[losort[kn]];    // -neighbour internal
            ++kn;
        }
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int bk = bndPerm[k];
        if (bndIsEmpty[bk]) continue;   // whatever bval holds there -- a flux is zero now, an interpolate is not (item 36d)
        s += bval[bk];   // +boundary
    }
    d[c] = s / V[c];
}


// ---- leastSquares gradient (OF leastSquaresGrad.C + leastSquaresVectors.C) ---------------------
// Two per-cell gathers, no atomics, the same owner/losort/bndPerm walk the Gauss gradient uses. Each
// cell needs only its OWN inverted dd tensor: OpenFOAM's pVectors use invDd[own] and its nVectors
// invDd[nei], so on the owner side of a face the cell reads its own, and on the neighbour side likewise.
//
// d, the cell-to-cell vector, is already in DeviceMesh: (Cf - C_own) - (Cf - C_nei) = C_nei - C_own, and
// a boundary face's fvPatch::delta() is the PATCH-NORMAL PROJECTION of (Cf - C_faceCell),
// `nHat*(nHat & (Cf() - Cn()))` (fvPatch.C) -- NOT the raw difference, which is the same vector only on
// an orthogonal mesh. dBnd carries the raw Cf - C (cellLimitedGrad needs that one), so the projection is
// taken here from Sf/magSf at the same face.
// fvPatch::delta() on an uncoupled patch: the component of Cf - Cn along the face normal. The host's
// lsBoundaryDelta (fvc.cu) is the same expression, so the two arms cannot drift.
BRAE_HD inline vector lsqBndDelta(
    scalar dx, scalar dy, scalar dz,
    scalar sfx, scalar sfy, scalar sfz,
    scalar mag)
{
    const vector nH{sfx / mag, sfy / mag, sfz / mag};
    const scalar p = nH.x * dx + nH.y * dy + nH.z * dz;
    return vector{p * nH.x, p * nH.y, p * nH.z};
}

__global__
void lsqInvDdKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ dBndX, const scalar* __restrict__ dBndY, const scalar* __restrict__ dBndZ,
    scalar* __restrict__ idd)     // 6*nC, component-major
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    symmTensor dd{0, 0, 0, 0, 0, 0};
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
    {
        const vector d{dOwnX[f] - dNeiX[f], dOwnY[f] - dNeiY[f], dOwnZ[f] - dNeiZ[f]};
        dd = dd + ((1.0 - w[f]) * (magSf[f] / magSqr(d))) * sqr(d);
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        const vector d{dOwnX[f] - dNeiX[f], dOwnY[f] - dNeiY[f], dOwnZ[f] - dNeiZ[f]};
        dd = dd + (w[f] * (magSf[f] / magSqr(d))) * sqr(d);
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int bk = bndPerm[k];
        if (bndIsEmpty[bk]) continue;   // emptyFvPatch::size() == 0 -- this is what leaves dd singular in 2-D
        const int gf = bndGFace[bk];
        const vector d = lsqBndDelta(dBndX[bk], dBndY[bk], dBndZ[bk], Sfx[gf], Sfy[gf], Sfz[gf], magSf[gf]);
        dd = dd + (magSf[gf] / magSqr(d)) * sqr(d);
    }
    const symmTensor r = safeInv(dd);
    idd[0 * nC + c] = r.xx; idd[1 * nC + c] = r.xy; idd[2 * nC + c] = r.xz;
    idd[3 * nC + c] = r.yy; idd[4 * nC + c] = r.yz; idd[5 * nC + c] = r.zz;
}

__global__
void lsqGradKernel(
    int nC,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ vf,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ dBndX, const scalar* __restrict__ dBndY, const scalar* __restrict__ dBndZ,
    const scalar* __restrict__ bval,
    const scalar* __restrict__ idd,
    scalar* __restrict__ gx, scalar* __restrict__ gy, scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const symmTensor iv{idd[0*nC+c], idd[1*nC+c], idd[2*nC+c], idd[3*nC+c], idd[4*nC+c], idd[5*nC+c]};
    const scalar vc = vf[c];
    vector s{0, 0, 0};
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
    {
        const vector d{dOwnX[f] - dNeiX[f], dOwnY[f] - dNeiY[f], dOwnZ[f] - dNeiZ[f]};
        const scalar msd = magSf[f] / magSqr(d);
        s += ((1.0 - w[f]) * msd * (vf[nei[f]] - vc)) * (iv & d);
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        const vector d{dOwnX[f] - dNeiX[f], dOwnY[f] - dNeiY[f], dOwnZ[f] - dNeiZ[f]};
        const scalar msd = magSf[f] / magSqr(d);
        // grad[nei] -= nVectors*dvf with nVectors = -w*msd*(invDd[nei] & d) and dvf = vf[nei] - vf[own]
        s += (w[f] * msd * (vc - vf[own[f]])) * (iv & d);
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int bk = bndPerm[k];
        if (bndIsEmpty[bk]) continue;
        const int gf = bndGFace[bk];
        const vector d = lsqBndDelta(dBndX[bk], dBndY[bk], dBndZ[bk], Sfx[gf], Sfy[gf], Sfz[gf], magSf[gf]);
        s += ((magSf[gf] / magSqr(d)) * (bval[bk] - vc)) * (iv & d);
    }
    // No division by V: the fit vectors already carry the normalisation.
    gx[c] = s.x; gy[c] = s.y; gz[c] = s.z;
}



__global__
void gradKernel(
    int nC,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ vol,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndGFace,
    const label* __restrict__ bndIsEmpty,   // emptyFvPatch::size() == 0: an empty face is never in OpenFOAM's sum
    const scalar* __restrict__ bval,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz,
    const int* __restrict__ skipIf)     // device flag: when set, this launch is a no-op (the grad(U) memo hit)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (skipIf && *skipIf) return;

    // A TRANSCRIPTION of the host's fvc::gaussGrad (fvc.cu), which is OpenFOAM's gaussGrad::gradf:
    //   the face value in OpenFOAM's arithmetic, lambda*(P - N) + N (surfaceInterpolationScheme.C:270),
    //   not w*P + (1 - w)*N -- on a face whose two cells hold the same value it returns that value
    //   exactly, and the other form can miss by an ulp whose SIGN is the gradient of a uniform field;
    //   and this cell's internal faces summed in FACE ORDER, as the host's face loop reaches them --
    //   the owner and neighbour lists are each ascending, so they are merged by face index. MEASURED
    //   before, on a sheared 960-cell box: 912 cells differed from the host in some bit on a smooth
    //   field, 934 on a uniform one, where the difference (3.9e-27) was the size of the gradient itself.
    //   tests/test_device_gauss_grad.cu holds the two to memcmp.
    scalar sx = 0.0, sy = 0.0, sz = 0.0;
    int fo = ownerStart[c];
    const int foEnd = ownerStart[c + 1];
    int kn = losortStart[c];
    const int knEnd = losortStart[c + 1];
    while (fo < foEnd || kn < knEnd)
    {
        const bool takeOwner = (kn >= knEnd) || (fo < foEnd && fo < losort[kn]);
        const int f = takeOwner ? fo : losort[kn];
        const scalar pf = w[f] * (vol[own[f]] - vol[nei[f]]) + vol[nei[f]];
        if (takeOwner)   // +owner internal
        {
            sx += Sfx[f] * pf;
            sy += Sfy[f] * pf;
            sz += Sfz[f] * pf;
            ++fo;
        }
        else             // -neighbour internal
        {
            sx -= Sfx[f] * pf;
            sy -= Sfy[f] * pf;
            sz -= Sfz[f] * pf;
            ++kn;
        }
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)   // +boundary
    {
        const int kk = bndPerm[k];
        // Skipped as the cellLimited kernel below skips them, and for the same reason: OpenFOAM
        // cannot sum a face of a zero-sized patch. Sf_x and Sf_y of an extruded empty face are
        // bitwise zero, so this changes nothing in-plane; g_z goes from the cancellation of two
        // opposite 1e+00-scale terms to internal-face round-off, which is what OpenFOAM has (item 36c).
        if (bndIsEmpty[kk]) continue;
        const int f = bndGFace[kk];
        const scalar pv = bval[kk];
        sx += Sfx[f] * pv;
        sy += Sfy[f] * pv;
        sz += Sfz[f] * pv;
    }
    gx[c] = sx / V[c];
    gy[c] = sy / V[c];
    gz[c] = sz / V[c];
}
} // namespace


void deviceInterpolate(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol, DeviceBuffer<scalar>& sfInt)
{
    sfInt.resize(dm.nInternalFaces);
    interpKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), dm.w.data(), vol.data(), sfInt.data());
    cudaCheck(cudaGetLastError(), "interp");
}


void deviceDiv(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& bval, DeviceBuffer<scalar>& d,
               const DeviceBuffer<scalar>* V)
{
    d.resize(dm.nCells);
    divKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                           phiInt.data(), dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(), bval.data(),
                                           V ? V->data() : dm.V.data(), d.data());
    cudaCheck(cudaGetLastError(), "div");
}


// THE SAME FIT, N FIELDS AT A TIME, IN ONE LAUNCH (FP-3). The least-squares twin of gradFusedKernel
// below and the same argument: fusing N independent fields is a loop interchange -- the same three
// face loops in the same order, each field's sum in its own registers, the same expression text as
// lsqGradKernel's so nvcc contracts the same multiply-adds -- and the shared operands (d, |Sf|/|d|^2
// and the cell's (invDd & d) vector) are READ, never reassociated. tests/test_lsq_grad_fused.cu holds
// every field to memcmp against lsqGradKernel, which stays as it is for exactly that purpose.
template<int N>
struct LsqFusedFields
{
    const scalar* vol[N];
    const scalar* bval[N];
    scalar* gx[N];
    scalar* gy[N];
    scalar* gz[N];
};

template<int N>
__global__
void lsqGradFusedKernel(
    int nC,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ dBndX, const scalar* __restrict__ dBndY, const scalar* __restrict__ dBndZ,
    const scalar* __restrict__ idd,
    LsqFusedFields<N> fld)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const symmTensor iv{idd[0*nC+c], idd[1*nC+c], idd[2*nC+c], idd[3*nC+c], idd[4*nC+c], idd[5*nC+c]};
    scalar vc[N];
    vector s[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        vc[i] = fld.vol[i][c];
        s[i] = vector{0, 0, 0};
    }
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
    {
        const vector d{dOwnX[f] - dNeiX[f], dOwnY[f] - dNeiY[f], dOwnZ[f] - dNeiZ[f]};
        const scalar msd = magSf[f] / magSqr(d);
        const label n = nei[f];
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            s[i] += ((1.0 - w[f]) * msd * (fld.vol[i][n] - vc[i])) * (iv & d);
        }
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        const vector d{dOwnX[f] - dNeiX[f], dOwnY[f] - dNeiY[f], dOwnZ[f] - dNeiZ[f]};
        const scalar msd = magSf[f] / magSqr(d);
        const label o = own[f];
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            s[i] += (w[f] * msd * (vc[i] - fld.vol[i][o])) * (iv & d);
        }
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int bk = bndPerm[k];
        if (bndIsEmpty[bk]) continue;
        const int gf = bndGFace[bk];
        const vector d = lsqBndDelta(dBndX[bk], dBndY[bk], dBndZ[bk], Sfx[gf], Sfy[gf], Sfz[gf], magSf[gf]);
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            s[i] += ((magSf[gf] / magSqr(d)) * (fld.bval[i][bk] - vc[i])) * (iv & d);
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        fld.gx[i][c] = s[i].x;
        fld.gy[i][c] = s[i].y;
        fld.gz[i][c] = s[i].z;
    }
}

namespace {
template<int N>
void launchLsqFused(
    const DeviceMesh& dm,
    const scalar* const* vol,
    const scalar* const* bval,
    scalar* const* gx,
    scalar* const* gy,
    scalar* const* gz)
{
    LsqFusedFields<N> fld;
    for (int i = 0; i < N; ++i)
    {
        fld.vol[i]  = vol[i];
        fld.bval[i] = bval[i];
        fld.gx[i]   = gx[i];
        fld.gy[i]   = gy[i];
        fld.gz[i]   = gz[i];
    }
    const int nC = dm.nCells;
    lsqGradFusedKernel<N><<<nBlocks(nC), TPB>>>(
        nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
        dm.w.data(), dm.magSf.data(),
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(),
        dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(), dm.bndGFace.data(),
        dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), lsqInvDdFor(dm), fld);
    cudaCheck(cudaGetLastError(), "lsqGradFused");
}
} // namespace


const scalar* lsqInvDdFor(const DeviceMesh& dm)
{
    // The control: rebuild at every request, which is what every gradient did before FP-3.
    static const bool recompute = []()
    {
        const char* e = std::getenv("BRAE_LSQ_INVDD");
        return e && std::string(e) == "recompute";
    }();
    const int nC = dm.nCells;
    const std::size_t want = static_cast<std::size_t>(6) * nC;
    if (dm.lsqInvDd.size() == want && !recompute) return dm.lsqInvDd.data();
    static bool announced = false;
    if (!announced)
    {
        announced = true;
        std::printf("  leastSquares: the inverted dd tensor is built once per mesh (FP-3); "
                    "BRAE_LSQ_INVDD=recompute rebuilds it at every gradient\n");
    }
    dm.lsqInvDd.resize(want);
    lsqInvDdKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), dm.w.data(), dm.magSf.data(),
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(),
        dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(), dm.bndGFace.data(),
        dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), dm.lsqInvDd.data());
    cudaCheck(cudaGetLastError(), "lsqInvDd");
    return dm.lsqInvDd.data();
}


void deviceLeastSquaresGrad(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol,
                            const DeviceBuffer<scalar>& bval,
                            DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz)
{
    const int nC = dm.nCells;
    gx.resize(nC); gy.resize(nC); gz.resize(nC);
    // The single-field kernel stays the reference the fused one is held against; it does NOT forward
    // to the fused path (the same reason deviceGaussGrad does not).
    lsqGradKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
        dm.w.data(), dm.magSf.data(),
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(),
        dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(), vol.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(), dm.bndGFace.data(),
        dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), bval.data(), lsqInvDdFor(dm),
        gx.data(), gy.data(), gz.data());
    cudaCheck(cudaGetLastError(), "lsqGrad");
}


void deviceLeastSquaresGradFusedRaw(const DeviceMesh& dm, int n,
                                    const scalar* const* vol, const scalar* const* bval,
                                    scalar* const* gx, scalar* const* gy, scalar* const* gz)
{
    // Refuse rather than truncate, as deviceGaussGradFused does.
    if (n < 1 || n > 3)
    {
        throw std::runtime_error("brae: deviceLeastSquaresGradFused takes 1, 2 or 3 fields, asked for "
                                 + std::to_string(n) + ".");
    }
    switch (n)
    {
        case 1:
            launchLsqFused<1>(dm, vol, bval, gx, gy, gz);
            break;
        case 2:
            launchLsqFused<2>(dm, vol, bval, gx, gy, gz);
            break;
        default:
            launchLsqFused<3>(dm, vol, bval, gx, gy, gz);
            break;
    }
}


void deviceLeastSquaresGradFused(const DeviceMesh& dm, int n,
                                 const DeviceBuffer<scalar>* const* vol, const DeviceBuffer<scalar>* const* bval,
                                 DeviceBuffer<scalar>* gx, DeviceBuffer<scalar>* gy, DeviceBuffer<scalar>* gz)
{
    if (n < 1 || n > 3)
    {
        throw std::runtime_error("brae: deviceLeastSquaresGradFused takes 1, 2 or 3 fields, asked for "
                                 + std::to_string(n) + ".");
    }
    const scalar* v[3] = {nullptr, nullptr, nullptr};
    const scalar* b[3] = {nullptr, nullptr, nullptr};
    scalar* x[3] = {nullptr, nullptr, nullptr};
    scalar* y[3] = {nullptr, nullptr, nullptr};
    scalar* z[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < n; ++i)
    {
        gx[i].resize(dm.nCells);
        gy[i].resize(dm.nCells);
        gz[i].resize(dm.nCells);
        v[i] = vol[i]->data();
        b[i] = bval[i]->data();
        x[i] = gx[i].data();
        y[i] = gy[i].data();
        z[i] = gz[i].data();
    }
    deviceLeastSquaresGradFusedRaw(dm, n, v, b, x, y, z);
}


void deviceGaussGrad(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& vol,
    const DeviceBuffer<scalar>& bval,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz,
    const int* skipIf)
{
    gx.resize(dm.nCells);
    gy.resize(dm.nCells);
    gz.resize(dm.nCells);
    gradKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), vol.data(),
                                            dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                            dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(), dm.bndIsEmpty.data(), bval.data(),
                                            dm.V.data(), gx.data(), gy.data(), gz.data(), skipIf);
    cudaCheck(cudaGetLastError(), "gaussGrad");
}


namespace {
// THE SAME GRADIENT, N FIELDS AT A TIME, IN ONE LAUNCH.
//
// WHY. nsys on the compressible iteration at 305,760 cells (12 outer iterations): gradKernel ran 16
// times per outer iteration at 281 us each -- 4.5 ms/it, 15% of all GPU work and second only to the
// matrix-vector product, in an iteration that is launch- and bandwidth-bound (about 30 ms of GPU-busy
// time in 914 launches). Nine of the sixteen are velocity components in groups of three, and each of
// those three re-reads the WHOLE of the addressing and geometry -- owner, nei, w, Sf*, ownerStart,
// losort, losortStart, bndCellStart, bndPerm, bndGFace, bndIsEmpty, V -- to carry one field, which is a
// small fraction of the traffic. Reading the row once is what took the colour Gauss-Seidel sweep from
// 524 to 231 us at 306k (device_colour_gauss_seidel.cu); this is the same lever on the assembly side.
//
// WHY IT IS BIT-IDENTICAL TO gradKernel, per field. Fusing N independent fields is a loop interchange
// and nothing else: the same three loops, in the same order, over the same faces, with the same
// expressions, each field accumulated into its own registers in the same sequence. Nothing is
// reassociated and nothing is shared but the operands that are read. tests/test_grad_fused.cu holds
// that to memcmp against N separate gradKernel launches, and its one-ulp control proves the per-field
// registers are not crossed -- which is the failure mode a fused kernel actually has.
template<int N>
struct GradFusedFields
{
    const scalar* vol[N];
    const scalar* bval[N];
    scalar* gx[N];
    scalar* gy[N];
    scalar* gz[N];
};


template<int N>
__global__
void gradFusedKernel(
    int nC,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndGFace,
    const label* __restrict__ bndIsEmpty,   // emptyFvPatch::size() == 0: an empty face is never in OpenFOAM's sum
    const scalar* __restrict__ V,
    GradFusedFields<N> fld,
    const int* __restrict__ skipIf)     // device flag: when set, this launch is a no-op (the grad(U) memo hit)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (skipIf && *skipIf) return;

    scalar sx[N], sy[N], sz[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        sx[i] = 0.0;
        sy[i] = 0.0;
        sz[i] = 0.0;
    }
    // gradKernel's transcription of the host's gaussGrad, per field: OpenFOAM's face arithmetic and the
    // internal faces merged into face order (see gradKernel)
    int fo = ownerStart[c];
    const int foEnd = ownerStart[c + 1];
    int kn = losortStart[c];
    const int knEnd = losortStart[c + 1];
    while (fo < foEnd || kn < knEnd)
    {
        const bool takeOwner = (kn >= knEnd) || (fo < foEnd && fo < losort[kn]);
        const int f = takeOwner ? fo : losort[kn];
        const scalar wf = w[f];
        const label o = own[f], n = nei[f];
        const scalar sfx = Sfx[f], sfy = Sfy[f], sfz = Sfz[f];
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            const scalar pf = wf * (fld.vol[i][o] - fld.vol[i][n]) + fld.vol[i][n];
            if (takeOwner)
            {
                sx[i] += sfx * pf;
                sy[i] += sfy * pf;
                sz[i] += sfz * pf;
            }
            else
            {
                sx[i] -= sfx * pf;
                sy[i] -= sfy * pf;
                sz[i] -= sfz * pf;
            }
        }
        if (takeOwner)
        {
            ++fo;
        }
        else
        {
            ++kn;
        }
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)   // +boundary
    {
        const int kk = bndPerm[k];
        // The same skip as gradKernel's, for the same reason: OpenFOAM cannot sum a face of a
        // zero-sized patch, so an empty face is in no surface sum (item 36c).
        if (bndIsEmpty[kk]) continue;
        const int f = bndGFace[kk];
        const scalar sfx = Sfx[f], sfy = Sfy[f], sfz = Sfz[f];
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            const scalar pv = fld.bval[i][kk];
            sx[i] += sfx * pv;
            sy[i] += sfy * pv;
            sz[i] += sfz * pv;
        }
    }
    const scalar Vc = V[c];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        fld.gx[i][c] = sx[i] / Vc;
        fld.gy[i][c] = sy[i] / Vc;
        fld.gz[i][c] = sz[i] / Vc;
    }
}


template<int N>
void launchGradFused(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>* const* vol,
    const DeviceBuffer<scalar>* const* bval,
    DeviceBuffer<scalar>* gx,
    DeviceBuffer<scalar>* gy,
    DeviceBuffer<scalar>* gz,
    const int* skipIf)
{
    GradFusedFields<N> fld;
    for (int i = 0; i < N; ++i)
    {
        fld.vol[i]  = vol[i]->data();
        fld.bval[i] = bval[i]->data();
        fld.gx[i]   = gx[i].data();
        fld.gy[i]   = gy[i].data();
        fld.gz[i]   = gz[i].data();
    }
    gradFusedKernel<N><<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                                    dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                                    dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                                    dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(), dm.bndIsEmpty.data(),
                                                    dm.V.data(), fld, skipIf);
    cudaCheck(cudaGetLastError(), "gaussGradFused");
}
} // namespace


void deviceGaussGradFused(
    const DeviceMesh& dm,
    int n,
    const DeviceBuffer<scalar>* const* vol,
    const DeviceBuffer<scalar>* const* bval,
    DeviceBuffer<scalar>* gx,
    DeviceBuffer<scalar>* gy,
    DeviceBuffer<scalar>* gz,
    const int* skipIf)
{
    // Refuse rather than truncate: silently gradient-ing the first three of four fields is exactly the
    // class of quiet substitution this project keeps finding.
    if (n < 1 || n > 3)
    {
        throw std::runtime_error("brae: deviceGaussGradFused takes 1, 2 or 3 fields, asked for "
                                 + std::to_string(n) + ".");
    }
    for (int i = 0; i < n; ++i)
    {
        gx[i].resize(dm.nCells);
        gy[i].resize(dm.nCells);
        gz[i].resize(dm.nCells);
    }
    switch (n)
    {
        case 1:
            launchGradFused<1>(dm, vol, bval, gx, gy, gz, skipIf);
            break;
        case 2:
            launchGradFused<2>(dm, vol, bval, gx, gy, gz, skipIf);
            break;
        default:
            launchGradFused<3>(dm, vol, bval, gx, gy, gz, skipIf);
            break;
    }
}


namespace {
// OF cellLimitedGrad<minmod>::limitFaceCmpt: r = maxDelta/extrapolate (or minDelta/extrapolate); minmod limiter
// min(r,1). |extrapolate|~0 -> face imposes no constraint (returns 1). SMALL = Foam::SMALL (1e-15, double).
__device__ __forceinline__
scalar limFace(scalar maxD, scalar minD, scalar ex)
{
    if (ex >  1e-15) return fmin(maxD / ex, 1.0);
    if (ex < -1e-15) return fmin(minD / ex, 1.0);
    return 1.0;
}


__global__
void cellLimitGradKernel(
    int nC,
    scalar k,
    const scalar* __restrict__ U,
    const scalar* __restrict__ Ubnd,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ owner,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ dBndX,
    const scalar* __restrict__ dBndY,
    const scalar* __restrict__ dBndZ,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar uc = U[c];
    scalar maxD = 0.0, minD = 0.0;   // incl. the cell itself (U[c]-U[c]=0)
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    {
        const scalar d = U[nei[f]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const scalar d = U[owner[losort[j]]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    // EMPTY PATCHES CONTRIBUTE NOTHING, because in OpenFOAM they cannot: emptyFvPatchField is a
    // ZERO-SIZED patch field (emptyFvPatchField.C:41), so there are no faces to evaluate. brae keeps
    // those faces in its addressing and has to skip them explicitly, and the host reference does
    // (cellLimitedGrad_cpp.cu:76 and :116). In the FACE loop it is not harmless: on a 2D mesh Cf - C for
    // an empty face points out of the plane, so the extrapolate is the out-of-plane gradient -- round-off
    // rather than physics -- and r = maxDelta/extrapolate is then a ratio of a real number to noise,
    // which can clamp the limiter far below what any real face asks for. Measured on the host when it
    // was fixed there: grad(U) 1.39e-02 -> 1.28e-14.
    //
    // LATENT ON EVERY REGISTERED FIXTURE, and saying so is the point. rho_ueqn_cuda_cell_limited is the
    // only arm that drives the limiter on a 2D mesh, and it passes at 2e-12 WITH the skip and WITHOUT
    // it. Both loops are inert on pitzDailyTurb for a reason worth writing down: an empty face's stored
    // value equals its cell's, so Ubnd - uc is 0 and cannot move a range that already includes the cell
    // itself at 0; and the mesh is axis-aligned, so dBnd is out-of-plane while the gradient is in it and
    // dBnd . g underflows to a limiter of 1. Neither holds in general -- the host's own measurement
    // above is what a case that breaks the second one costs -- so this brings the device into line with
    // the host and with OpenFOAM's zero-sized empty patch rather than fixing an observed number.
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const int bk = bndPerm[j];
        if (bndIsEmpty[bk]) continue;
        const scalar d = Ubnd[bk] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    if (k < 1.0)   // OF k<1 widening
    {
        const scalar wdn = (1.0/k - 1.0)*(maxD - minD);
        maxD += wdn;
        minD -= wdn;
    }
    const scalar gcx = gx[c], gcy = gy[c], gcz = gz[c];
    scalar lim = 1.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        lim = fmin(lim, limFace(maxD, minD, dOwnX[f]*gcx + dOwnY[f]*gcy + dOwnZ[f]*gcz));
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const int f = losort[j];
        lim = fmin(lim, limFace(maxD, minD, dNeiX[f]*gcx + dNeiY[f]*gcy + dNeiZ[f]*gcz));
    }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const int bk = bndPerm[j];
        if (bndIsEmpty[bk]) continue;   // see the note above -- here it is NOT round-off-harmless
        lim = fmin(lim, limFace(maxD, minD, dBndX[bk]*gcx + dBndY[bk]*gcy + dBndZ[bk]*gcz));
    }
    gx[c] = gcx*lim;
    gy[c] = gcy*lim;
    gz[c] = gcz*lim;
}

// ---- the coupled-patch path (see CellLimitInterface) -------------------------------------------------
// max/min and the limiter are all order-independent reductions, so scattering them from the faces with
// atomics gives the same answer every run whatever order the faces land in.
__device__ __forceinline__ void atomicFmax(scalar* a, scalar v)
{
    unsigned long long* p = reinterpret_cast<unsigned long long*>(a);
    unsigned long long old = *p, assumed;
    while (v > __longlong_as_double(old))
    {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(v));
        if (old == assumed) break;
    }
}
__device__ __forceinline__ void atomicFmin(scalar* a, scalar v)
{
    unsigned long long* p = reinterpret_cast<unsigned long long*>(a);
    unsigned long long old = *p, assumed;
    while (v < __longlong_as_double(old))
    {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(v));
        if (old == assumed) break;
    }
}

// Phase 1 on cells: maxD/minD from the internal + non-coupled boundary faces only. Split out of
// cellLimitGradKernel so the interface faces can be folded in between the two halves.
__global__
void cellLimitMinMaxKernel(
    int nC,
    const scalar* __restrict__ U,
    const scalar* __restrict__ Ubnd,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ owner,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    scalar* __restrict__ maxD,
    scalar* __restrict__ minD)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar uc = U[c];
    scalar mx = 0.0, mn = 0.0;   // incl. the cell itself (U[c]-U[c]=0)
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    { const scalar d = U[nei[f]] - uc; mx = fmax(mx,d); mn = fmin(mn,d); }
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    { const scalar d = U[owner[losort[j]]] - uc; mx = fmax(mx,d); mn = fmin(mn,d); }
    // Empty patches contribute nothing -- emptyFvPatchField is zero-sized in OpenFOAM, so these faces
    // do not exist there. Same skip as the single-kernel path, and the host reference at
    // cellLimitedGrad_cpp.cu:76.
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    { const int bk = bndPerm[j]; if (bndIsEmpty[bk]) continue;
      const scalar d = Ubnd[bk] - uc; mx = fmax(mx,d); mn = fmin(mn,d); }
    maxD[c] = mx;
    minD[c] = mn;
}

// Phase 2 on interface faces: the coupled neighbour joins the cell's range.
__global__
void ifMinMaxKernel(int n, const label* __restrict__ own, const scalar* __restrict__ nbrVal,
                    const scalar* __restrict__ U, scalar* __restrict__ maxD, scalar* __restrict__ minD)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = own[i];
    const scalar d = nbrVal[i] - U[c];
    atomicFmax(&maxD[c], d);
    atomicFmin(&minD[c], d);
}

// Phase 3: OF's k<1 widening, applied once the range is complete (interfaces included).
__global__
void widenKernel(int nC, scalar k, scalar* __restrict__ maxD, scalar* __restrict__ minD)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar wdn = (1.0/k - 1.0)*(maxD[c] - minD[c]);
    maxD[c] += wdn;
    minD[c] -= wdn;
}

// Phase 4 on cells: the limiter from the internal + non-coupled boundary face extrapolations.
__global__
void cellLimitFactorKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ dBndX, const scalar* __restrict__ dBndY, const scalar* __restrict__ dBndZ,
    const scalar* __restrict__ maxD, const scalar* __restrict__ minD,
    const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
    scalar* __restrict__ lim)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar gcx = gx[c], gcy = gy[c], gcz = gz[c], mx = maxD[c], mn = minD[c];
    scalar l = 1.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        l = fmin(l, limFace(mx, mn, dOwnX[f]*gcx + dOwnY[f]*gcy + dOwnZ[f]*gcz));
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    { const int f = losort[j]; l = fmin(l, limFace(mx, mn, dNeiX[f]*gcx + dNeiY[f]*gcy + dNeiZ[f]*gcz)); }
    // ...and here the skip is load-bearing rather than tidy: Cf - C on an empty face points out of the
    // 2D plane, so this extrapolate is round-off and the ratio against maxDelta can clamp the limiter far
    // below what any real face asks for (cellLimitedGrad_cpp.cu:111-116).
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    { const int bk = bndPerm[j]; if (bndIsEmpty[bk]) continue;
      l = fmin(l, limFace(mx, mn, dBndX[bk]*gcx + dBndY[bk]*gcy + dBndZ[bk]*gcz)); }
    lim[c] = l;
}

// Phase 5 on interface faces: OF's second loop is over EVERY boundary face, not just the uncoupled ones.
__global__
void ifLimitKernel(int n, const label* __restrict__ own,
                   const scalar* __restrict__ dx, const scalar* __restrict__ dy, const scalar* __restrict__ dz,
                   const scalar* __restrict__ maxD, const scalar* __restrict__ minD,
                   const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
                   scalar* __restrict__ lim)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = own[i];
    atomicFmin(&lim[c], limFace(maxD[c], minD[c], dx[i]*gx[c] + dy[i]*gy[c] + dz[i]*gz[c]));
}

__global__
void applyLimitKernel(int nC, const scalar* __restrict__ lim,
                      scalar* __restrict__ gx, scalar* __restrict__ gy, scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar l = lim[c];
    gx[c] *= l; gy[c] *= l; gz[c] *= l;
}
} // namespace


namespace {
// THE SAME LIMITER, N FIELDS AT A TIME, IN ONE LAUNCH (FP-4).
//
// WHY. nsys on gasMixing/injectorPipe at 74,650 cells (--cuda-graph-trace=node, 20 iterations, after
// FP-3): cellLimitGradKernel is 12 launches and 1.90 of the iteration's 13.73 GPU ms, and 9 of those
// launches -- 1.41 ms -- are inside the momentum phase, which is 44% of it. They are the three velocity
// components at three sites (the limitedLinearV weights, the corrected snGrad's gradient and
// linearUpwindV's), each launch re-reading the WHOLE of the addressing and the face offsets -- owner,
// nei, losort, losortStart, ownerStart, bndCellStart, bndPerm, bndIsEmpty, dOwn*, dNei*, dBnd* -- and
// walking every face of every cell SIX times (three loops for the range, three for the limiter) to
// carry one component. Reading that row once for three fields is the lever that worked on the Gauss
// gradient (gradFusedKernel) and on the leastSquares fit (lsqGradFusedKernel).
//
// WHY IT IS BIT-IDENTICAL, per field. Fusing N independent fields is a loop interchange: the same faces
// in the same order, the same expressions, each field's range, limiter and result in its own registers.
// Nothing is reassociated; the shared operands (the offsets, the addressing) are read, never combined
// across fields. min and max over the faces are per field, so no cross-field reduction exists to get
// wrong. tests/test_cell_limit_grad_fused.cu holds every field to memcmp against deviceCellLimitGrad,
// which keeps its own kernel for exactly that purpose.
//
// WHAT THIS ONE WAS WORTH, MEASURED (injectorPipe, 3 runs of 100 iterations and an nsys profile): 12
// limiter launches per iteration became 8 and their GPU time 1.90 -> 1.74 ms, and the ITERATION did not
// move outside run-to-run noise. The limiter's cost is its SIX face-loop passes over scattered
// neighbour values, and fusing N fields removes no pass at all -- only the re-reads of the addressing.
// Occupancy was ruled out too: 92 registers against the single-field kernel's 58, and capping it to 80
// with __launch_bounds__(TPB, 3) moved 0.780 ms to 0.771. The pass count is what matters, which is what
// the pass count is what matters -- and THAT WAS MEASURED TOO, and it is not the answer either. A
// kernel that fused the gradient's three face passes with the limiter's range pass (they read the same
// values at the same faces, so nine passes become six) was written, held bit-identical to
// gradient-then-limiter, and measured on the same case: the two sites it served went 1.026 -> 0.999 ms
// per iteration, 0.04 of the iteration's 13.7. The second pass was already L2-resident, so sharing it
// bought the loop overhead and nothing else, and the kernel was reverted rather than kept as 200 lines
// of duplicated arithmetic that must stay bit-identical to two others forever. What is left on this row
// is the face-gather traffic itself, which is the scheme's own cost: on injectorPipe the momentum phase
// is 4.1 ms/it limited, 2.7 with grad(U) unlimited and 2.5 with upwind divergence as well.
template<int N>
struct LimitFusedFields
{
    const scalar* U[N];
    const scalar* Ubnd[N];
    scalar* gx[N];
    scalar* gy[N];
    scalar* gz[N];
};

template<int N>
__global__
void cellLimitGradFusedKernel(
    int nC,
    scalar k,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ owner,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ dBndX, const scalar* __restrict__ dBndY, const scalar* __restrict__ dBndZ,
    LimitFusedFields<N> fld)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar uc[N], maxD[N], minD[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        uc[i] = fld.U[i][c];
        maxD[i] = 0.0;               // incl. the cell itself (U[c]-U[c]=0)
        minD[i] = 0.0;
    }
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    {
        const label n = nei[f];
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            const scalar d = fld.U[i][n] - uc[i];
            maxD[i] = fmax(maxD[i],d);
            minD[i] = fmin(minD[i],d);
        }
    }
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const label o = owner[losort[j]];
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            const scalar d = fld.U[i][o] - uc[i];
            maxD[i] = fmax(maxD[i],d);
            minD[i] = fmin(minD[i],d);
        }
    }
    // The empty-patch skip, for the reason cellLimitGradKernel spells out above: in the FACE loop below
    // it is load-bearing, not tidy.
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const int bk = bndPerm[j];
        if (bndIsEmpty[bk]) continue;
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            const scalar d = fld.Ubnd[i][bk] - uc[i];
            maxD[i] = fmax(maxD[i],d);
            minD[i] = fmin(minD[i],d);
        }
    }
    if (k < 1.0)   // OF k<1 widening
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            const scalar wdn = (1.0/k - 1.0)*(maxD[i] - minD[i]);
            maxD[i] += wdn;
            minD[i] -= wdn;
        }
    }
    scalar gcx[N], gcy[N], gcz[N], lim[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        gcx[i] = fld.gx[i][c];
        gcy[i] = fld.gy[i][c];
        gcz[i] = fld.gz[i][c];
        lim[i] = 1.0;
    }
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    {
        const scalar ax = dOwnX[f], ay = dOwnY[f], az = dOwnZ[f];
#pragma unroll
        for (int i = 0; i < N; ++i)
            lim[i] = fmin(lim[i], limFace(maxD[i], minD[i], ax*gcx[i] + ay*gcy[i] + az*gcz[i]));
    }
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const int f = losort[j];
        const scalar ax = dNeiX[f], ay = dNeiY[f], az = dNeiZ[f];
#pragma unroll
        for (int i = 0; i < N; ++i)
            lim[i] = fmin(lim[i], limFace(maxD[i], minD[i], ax*gcx[i] + ay*gcy[i] + az*gcz[i]));
    }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const int bk = bndPerm[j];
        if (bndIsEmpty[bk]) continue;
        const scalar ax = dBndX[bk], ay = dBndY[bk], az = dBndZ[bk];
#pragma unroll
        for (int i = 0; i < N; ++i)
            lim[i] = fmin(lim[i], limFace(maxD[i], minD[i], ax*gcx[i] + ay*gcy[i] + az*gcz[i]));
    }
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        fld.gx[i][c] = gcx[i]*lim[i];
        fld.gy[i][c] = gcy[i]*lim[i];
        fld.gz[i][c] = gcz[i]*lim[i];
    }
}

template<int N>
void launchLimitFused(
    const DeviceMesh& dm,
    scalar k,
    const DeviceBuffer<scalar>* const* U,
    const DeviceBuffer<scalar>* const* Ubnd,
    DeviceBuffer<scalar>* gx,
    DeviceBuffer<scalar>* gy,
    DeviceBuffer<scalar>* gz)
{
    LimitFusedFields<N> fld;
    for (int i = 0; i < N; ++i)
    {
        fld.U[i]    = U[i]->data();
        fld.Ubnd[i] = Ubnd[i]->data();
        fld.gx[i]   = gx[i].data();
        fld.gy[i]   = gy[i].data();
        fld.gz[i]   = gz[i].data();
    }
    cellLimitGradFusedKernel<N><<<nBlocks(dm.nCells), TPB>>>(dm.nCells, k,
        dm.ownerStart.data(), dm.nei.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), fld);
    cudaCheck(cudaGetLastError(), "cellLimitGradFused");
}
} // namespace


// The N-field form of the limiter below, for the callers that limit the three velocity components with
// one coefficient. Coupled interfaces are NOT handled here (they need the five-phase scatter, which is
// what deviceCellLimitGrad still does per field), so a caller with interfaces keeps the loop.
void deviceCellLimitGradFused(
    const DeviceMesh& dm,
    int n,
    const DeviceBuffer<scalar>* const* U,
    const DeviceBuffer<scalar>* const* Ubnd,
    DeviceBuffer<scalar>* gx,
    DeviceBuffer<scalar>* gy,
    DeviceBuffer<scalar>* gz,
    scalar k)
{
    // Refuse rather than truncate, as deviceGaussGradFused does: silently limiting the first three of
    // four fields is the class of quiet substitution this project keeps finding.
    if (n < 1 || n > 3)
    {
        throw std::runtime_error("brae: deviceCellLimitGradFused takes 1, 2 or 3 fields, asked for "
                                 + std::to_string(n) + ".");
    }
    switch (n)
    {
        case 1:  launchLimitFused<1>(dm, k, U, Ubnd, gx, gy, gz); break;
        case 2:  launchLimitFused<2>(dm, k, U, Ubnd, gx, gy, gz); break;
        default: launchLimitFused<3>(dm, k, U, Ubnd, gx, gy, gz); break;
    }
}


// cellLimited Gauss linear <k> (OF cellLimitedGrad<vector,minmod>), applied to ONE component's gradient: scale grad
// so the reconstructed face value C[c] + (Cf-C).grad stays within the cell's face-neighbour min/max. Per-component
// (caller loops the 3 U components). k=1 = full limiting (motorBike); k<1 widens the bounds (less limiting).
void deviceCellLimitGrad(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& U,
    const DeviceBuffer<scalar>& Ubnd,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz,
    scalar k,
    const CellLimitInterface* ifs,
    int nIfs)
{
    int nIfFaces = 0;
    for (int i = 0; i < nIfs; ++i) if (ifs[i].n > 0 && ifs[i].nbrVal) nIfFaces += ifs[i].n;
    if (nIfFaces > 0)
    {
        // The five-phase form: the interface faces have to enter the range BEFORE the widening and the
        // limiter, and the extrapolation limit AFTER the range is complete. Same arithmetic as the
        // single-kernel path below, just split where a scatter can be inserted.
        const int nC = dm.nCells;
        DeviceBuffer<scalar> maxD(nC), minD(nC), lim(nC);
        cellLimitMinMaxKernel<<<nBlocks(nC), TPB>>>(nC, U.data(), Ubnd.data(),
            dm.ownerStart.data(), dm.nei.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
            dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(), maxD.data(), minD.data());
        for (int i = 0; i < nIfs; ++i)
            if (ifs[i].n > 0 && ifs[i].nbrVal)
                ifMinMaxKernel<<<nBlocks(ifs[i].n), TPB>>>(ifs[i].n, ifs[i].ownCell, ifs[i].nbrVal,
                                                           U.data(), maxD.data(), minD.data());
        if (k < 1.0) widenKernel<<<nBlocks(nC), TPB>>>(nC, k, maxD.data(), minD.data());
        cellLimitFactorKernel<<<nBlocks(nC), TPB>>>(nC,
            dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
            dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(),
            dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
            dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), maxD.data(), minD.data(),
            gx.data(), gy.data(), gz.data(), lim.data());
        for (int i = 0; i < nIfs; ++i)
            if (ifs[i].n > 0 && ifs[i].nbrVal)
                ifLimitKernel<<<nBlocks(ifs[i].n), TPB>>>(ifs[i].n, ifs[i].ownCell,
                    ifs[i].dOwnX, ifs[i].dOwnY, ifs[i].dOwnZ, maxD.data(), minD.data(),
                    gx.data(), gy.data(), gz.data(), lim.data());
        applyLimitKernel<<<nBlocks(nC), TPB>>>(nC, lim.data(), gx.data(), gy.data(), gz.data());
        cudaCheck(cudaGetLastError(), "cellLimitGradInterface");
        return;
    }
    cellLimitGradKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, k, U.data(), Ubnd.data(),
        dm.ownerStart.data(), dm.nei.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), gx.data(), gy.data(), gz.data());
    cudaCheck(cudaGetLastError(), "cellLimitGrad");
}

} // namespace brae
