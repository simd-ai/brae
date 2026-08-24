// cf GPU offload -- per-iteration flow BC value updates (updateCoeffs): the BC state that changes with the flux/
// velocity each SIMPLE iteration -- inletOutlet/outletInlet, mixed freestream, pressureInletOutletVelocity,
// slip/symmetry, totalPressure, and the HbyA constraints at slip/mixed faces. Split from device_boundary.cu
// (the BC matrix/flux contributions are in device_boundary_assembly.cu). Shared decls: device_boundary.cuh.
#include "device_boundary.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


// inletOutlet/outletInlet updateCoeffs: valueFraction is BINARY from the flux sign. inletOutlet -> inflow
// (phi<0) = fixedValue; outletInlet (freestreamPressure) -> the OPPOSITE: outflow (phi>=0) = fixedValue.
__device__
void ioUpdateKernel(
    int n,
    const label* __restrict__ ioMask,
    const label* __restrict__ oioMask,
    const scalar* __restrict__ phiB,
    label* __restrict__ bcType)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (ioMask[i])       bcType[i] = (phiB[i] <  0.0) ? 1 : 0;            // inflow = fixedValue(inletValue)
    else if (oioMask[i]) bcType[i] = (phiB[i] >= 0.0) ? 1 : 0;            // outflow = fixedValue(outletValue)
}


// mixed freestream updateCoeffs: vf = 0.5 -/+ 0.5*(U.n)/|U|. The normal flux U.n = phi_b/|Sf| is exact; |U| is the
// LOCAL adjacent-cell speed (OF uses the patch |U|; the cell value ~= it and avoids the vf circularity). Using the
// freestream |Uinf| instead is fine at a true far field but mis-scales an outlet whose speed != |Uinf|. Continuous
// in the flow angle (not a binary switch); at grazing faces (phi~0) vf->0.5 (the Robin midpoint OF uses).
__device__
void mixedUpdateKernel(
    int n,
    const label* __restrict__ maskU,
    const label* __restrict__ maskP,
    const label* __restrict__ fc,
    const scalar* __restrict__ phiB,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ rhoBnd,   // compressible: phiB is a MASS flux, so U.n = phiB/(rho_b*|Sf|)
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ Ub0,     // EVALUATED patch velocity (OF's `Up`), not the cell value
    const scalar* __restrict__ Ub1,
    const scalar* __restrict__ Ub2,
    const scalar* __restrict__ nx,      // unit face normal
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    scalar* __restrict__ vfU0,
    scalar* __restrict__ vfU1,
    scalar* __restrict__ vfU2,
    scalar* __restrict__ vfP)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const bool mu = maskU[i], mp = maskP[i];
    if (!mu && !mp) return;
    // OF, exactly:
    //   freestreamVelocity  valueFraction = 0.5 - 0.5*(Up & nf)/mag(Up)   (…VelocityFvPatchVectorField.C:106)
    //   freestreamPressure  valueFraction = 0.5 + 0.5*(Up & nf)/mag(Up)   (…PressureFvPatchScalarField.C:119)
    // where Up is the PATCH velocity -- the same vector in both the dot product and the magnitude, so the
    // ratio is a genuine direction cosine in [-1,1].
    //
    // brae previously formed it from mixed quantities: the normal component came from the face FLUX and
    // the magnitude from the ADJACENT CELL speed. Those disagree wherever the patch and cell velocities
    // differ, so the ratio was not a cosine, needed clamping to stay in range, and drifted from OF's
    // continuous Robin blend at exactly the angles the blend exists to handle.
    const scalar ubx = Ub0[i], uby = Ub1[i], ubz = Ub2[i];
    const scalar mUb = sqrt(ubx*ubx + uby*uby + ubz*ubz);
    scalar ct = (ubx*nx[i] + uby*ny[i] + ubz*nz[i]) / fmax(mUb, 1e-30);
    ct = fmin(fmax(ct, -1.0), 1.0);   // |cos| <= 1 analytically; the clamp is only against roundoff now
    if (mu)   // velocity sign
    {
        const scalar vfu = 0.5 - 0.5 * ct;
        vfU0[i] = vfu;
        vfU1[i] = vfu;
        vfU2[i] = vfu;
    }
    if (mp) vfP[i] = 0.5 + 0.5 * ct;                                                              // pressure sign
}


// pressureInletOutletVelocity updateCoeffs (directionMixed): per piov face, outflow -> zeroGradient (bcType 0),
// inflow -> fixedValue (bcType 1) with refValue = n*(n.U_cell) (the normal projection; tangential refValue 0).
__device__
void piovUpdateKernel(
    int n,
    const label* __restrict__ piov,
    const label* __restrict__ fc,
    const scalar* __restrict__ phiB,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    label* __restrict__ ty0,
    label* __restrict__ ty1,
    label* __restrict__ ty2,
    scalar* __restrict__ r0,
    scalar* __restrict__ r1,
    scalar* __restrict__ r2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !piov[i]) return;

    if (phiB[i] >= 0.0)   // outflow -> zeroGradient
    {
        ty0[i] = ty1[i] = ty2[i] = 0;
    }
    else   // inflow -> fixedValue normal projection
    {
        const int c = fc[i];
        const scalar Un = nx[i] * Ux[c] + ny[i] * Uy[c] + nz[i] * Uz[c];         // n . U_cell
        ty0[i] = ty1[i] = ty2[i] = 1;
        r0[i] = nx[i] * Un;
        r1[i] = ny[i] * Un;
        r2[i] = nz[i] * Un;
    }
}


// slip/symmetry updateCoeffs (OF basicSymmetry, general normal): per symMask face, per component k set the mixed
// valueFraction vf_k = |n_k| and ref_k = U_c[k] - sign(n_k)*(n.U_c). The cat-5 kernels then give valueIC_k = 1-|n_k|,
// gradIC_k = -dc*|n_k|, value = U_c - n(n.U_c), OF's symmetry coeffs. sign(0)=0 -> tangential (n_k=0) is zeroGradient.
__device__
void symUpdateKernel(
    int n,
    const label* __restrict__ sym,
    const label* __restrict__ fc,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ vf0,
    scalar* __restrict__ vf1,
    scalar* __restrict__ vf2,
    scalar* __restrict__ r0,
    scalar* __restrict__ r1,
    scalar* __restrict__ r2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !sym[i]) return;

    const int c = fc[i];
    const scalar nU = nx[i] * Ux[c] + ny[i] * Uy[c] + nz[i] * Uz[c];               // n . U_cell
    const scalar sgx = (nx[i] > 0) - (nx[i] < 0), sgy = (ny[i] > 0) - (ny[i] < 0), sgz = (nz[i] > 0) - (nz[i] < 0);
    vf0[i] = fabs(nx[i]);
    vf1[i] = fabs(ny[i]);
    vf2[i] = fabs(nz[i]);
    r0[i] = Ux[c] - sgx * nU;
    r1[i] = Uy[c] - sgy * nU;
    r2[i] = Uz[c] - sgz * nU;
}


// constrainHbyA at slip/symmetry faces: the wall flux must be 0 (no penetration), so HbyA_b = HbyA_c - n(n.HbyA_c)
// (tangential projection) -> phiHbyA_b = HbyA_b.Sf = |Sf|(HbyA_b.n) = 0. The cat-5 deviceBCValue blends `ref` (built
// from U, not HbyA), which is NOT the HbyA projection on an angled wall, so override it here. (Axis-aligned already
// gets 0 from ref_normal=0, so this is a no-op there.)
__device__
void symHbyAKernel(
    int n,
    const label* __restrict__ sym,
    const label* __restrict__ fc,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    const scalar* __restrict__ Hx,
    const scalar* __restrict__ Hy,
    const scalar* __restrict__ Hz,
    scalar* __restrict__ hxb,
    scalar* __restrict__ hyb,
    scalar* __restrict__ hzb)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !sym[i]) return;

    const int c = fc[i];
    const scalar nH = nx[i] * Hx[c] + ny[i] * Hy[c] + nz[i] * Hz[c];
    hxb[i] = Hx[c] - nx[i] * nH;
    hyb[i] = Hy[c] - ny[i] * nH;
    hzb[i] = Hz[c] - nz[i] * nH;
}


// at mixed faces, overwrite the HbyA boundary value with the U boundary value (constrainHbyA at fixesValue patches).
__device__
void selectMixedKernel(
    int n,
    const label* __restrict__ mask,
    const scalar* __restrict__ ux,
    const scalar* __restrict__ uy,
    const scalar* __restrict__ uz,
    scalar* __restrict__ hx,
    scalar* __restrict__ hy,
    scalar* __restrict__ hz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) { hx[i] = ux[i]; hy[i] = uy[i]; hz[i] = uz[i]; }
}


// at inletOutlet faces, put back the zero-gradient (extrapolated) HbyA -- see deviceExtrapolateIOHbyA.
__device__
void selectIOKernel(
    int n,
    const label* __restrict__ ioMask,
    const scalar* __restrict__ ex,
    const scalar* __restrict__ ey,
    const scalar* __restrict__ ez,
    scalar* __restrict__ hx,
    scalar* __restrict__ hy,
    scalar* __restrict__ hz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && ioMask[i]) { hx[i] = ex[i]; hy[i] = ey[i]; hz[i] = ez[i]; }
}


// totalPressure (OF totalPressureFvPatchScalarField::updateCoeffs). OF branches on the DIMENSIONS of p:
//
//   kinematic p (incompressible):  p = p0 - 0.5*neg(phi)*magSqr(U)
//   absolute p, psi "none":        p = p0 - 0.5*RHO*neg(phi)*magSqr(U)      <- rhoBnd supplies the rho
//
// The rho factor is not optional on a compressible case: it is the difference between a dynamic head in
// m2/s2 and one in Pa. Passing rhoBnd = null gives the incompressible form, bit-identical to before.
// (OF's third branch, the high-speed isentropic form with a named psi and gamma, is NOT implemented --
// readThermoCoeffs-style refusal happens at load rather than silently running the low-speed form.)
__device__
void tpUpdateKernel(
    int n,
    const label* __restrict__ tpMask,
    const scalar* __restrict__ p0,
    const scalar* __restrict__ phiB,
    const scalar* __restrict__ Uxb,
    const scalar* __restrict__ Uyb,
    const scalar* __restrict__ Uzb,
    const scalar* __restrict__ rhoBnd,   // compressible: rho at the face; null -> kinematic (incompressible)
    scalar* __restrict__ refValue)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !tpMask[i]) return;

    const scalar u2 = Uxb[i]*Uxb[i] + Uyb[i]*Uyb[i] + Uzb[i]*Uzb[i];
    const scalar rw = rhoBnd ? rhoBnd[i] : scalar(1);
    refValue[i] = p0[i] - 0.5 * rw * (phiB[i] < 0.0 ? 1.0 : 0.0) * u2;   // neg(phi)=inflow -> static = total - dynamic head
}
} // namespace


// flowRateInletVelocity, massFlowRate form (OF flowRateInletVelocityFvPatchVectorField::updateValues,
// extrapolateProfile false):  avgU = -flowRate/gSum(rho*magSf);  U_b = avgU*n.
//
// avgU is a single patch-wide scalar, computed by the caller as -mdot/dot(rhoBnd, maskedMagSf) -- the
// mask makes that dot product exactly OF's gSum over this patch. Faces outside the patch have mask 0
// and are left untouched.
__device__
void frUpdateKernel(
    int n,
    const scalar* __restrict__ mask,
    scalar avgU,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    scalar* __restrict__ refX,
    scalar* __restrict__ refY,
    scalar* __restrict__ refZ)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || mask[i] <= scalar(0)) return;
    refX[i] = avgU * nx[i];
    refY[i] = avgU * ny[i];
    refZ[i] = avgU * nz[i];
}

void deviceUpdateFlowRateInlet(
    DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& maskMagSf,
    scalar avgU,
    const DeviceBuffer<scalar>& nx,
    const DeviceBuffer<scalar>& ny,
    const DeviceBuffer<scalar>& nz)
{
    const int n = dbU.comp[0].n;
    if (n == 0) return;
    const scalar* maskD = maskMagSf.data(); const scalar* nxd = nx.data(); const scalar* nyd = ny.data(); const scalar* nzd = nz.data();
    scalar* r0 = dbU.comp[0].refValue.data(); scalar* r1 = dbU.comp[1].refValue.data(); scalar* r2 = dbU.comp[2].refValue.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
        frUpdateKernel(n, maskD, avgU, nxd, nyd, nzd, r0, r1, r2); });
    cudaCheck(cudaGetLastError(), "frUpdate");
}

void deviceUpdateInletOutlet(DeviceBoundary& db, const DeviceBuffer<scalar>& phiBnd)
{
    if (db.n == 0) return;
    const int n = db.n;
    const label* ioMask = db.ioMask.data(); const label* oioMask = db.oioMask.data();
    const scalar* phiBd = phiBnd.data(); label* bcType = db.bcType.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
        ioUpdateKernel(n, ioMask, oioMask, phiBd, bcType); });
    cudaCheck(cudaGetLastError(), "ioUpdate");
}


void deviceUpdateMixedFreestream(
    DeviceVectorBoundary& dbU,
    DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>* rhoBnd)
{
    const int n = dbP.n;
    if (n == 0) return;
    // OF's `Up` is the patch field's CURRENT value, i.e. the previous evaluate -- so evaluating here with
    // the existing valueFraction before overwriting it is the same lag OF has.
    DeviceBuffer<scalar> ub0, ub1, ub2;
    deviceBCValue(dbU.comp[0], Ux, ub0);
    deviceBCValue(dbU.comp[1], Uy, ub1);
    deviceBCValue(dbU.comp[2], Uz, ub2);
    {
        const label* maskU = dbU.comp[0].mixedMask.data(); const label* maskP = dbP.mixedMask.data();
        const label* fc = dbP.faceCell.data(); const scalar* phiBd = phiBnd.data(); const scalar* magSf = dbP.magSf.data();
        const scalar* rhoBndD = (rhoBnd && rhoBnd->size() == static_cast<std::size_t>(n)) ? rhoBnd->data() : nullptr;
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        const scalar* ub0d = ub0.data(); const scalar* ub1d = ub1.data(); const scalar* ub2d = ub2.data();
        const scalar* nxd = dbU.nx.data(); const scalar* nyd = dbU.ny.data(); const scalar* nzd = dbU.nz.data();
        scalar* vfU0 = dbU.comp[0].valueFraction.data(); scalar* vfU1 = dbU.comp[1].valueFraction.data();
        scalar* vfU2 = dbU.comp[2].valueFraction.data(); scalar* vfP = dbP.valueFraction.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            mixedUpdateKernel(n, maskU, maskP, fc, phiBd, magSf, rhoBndD, Uxd, Uyd, Uzd,
                               ub0d, ub1d, ub2d, nxd, nyd, nzd, vfU0, vfU1, vfU2, vfP); });
    }
    cudaCheck(cudaGetLastError(), "mixedUpdate");
}


void deviceUpdatePressureInletOutletVelocity(
    DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz)
{
    const int n = dbU.n;
    if (n == 0) return;
    {
        const label* piov = dbU.comp[0].piovMask.data(); const label* fc = dbU.comp[0].faceCell.data();
        const scalar* phiBd = phiBnd.data();
        const scalar* nxd = dbU.nx.data(); const scalar* nyd = dbU.ny.data(); const scalar* nzd = dbU.nz.data();
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        label* ty0 = dbU.comp[0].bcType.data(); label* ty1 = dbU.comp[1].bcType.data(); label* ty2 = dbU.comp[2].bcType.data();
        scalar* r0 = dbU.comp[0].refValue.data(); scalar* r1 = dbU.comp[1].refValue.data(); scalar* r2 = dbU.comp[2].refValue.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            piovUpdateKernel(n, piov, fc, phiBd, nxd, nyd, nzd, Uxd, Uyd, Uzd, ty0, ty1, ty2, r0, r1, r2); });
    }
    cudaCheck(cudaGetLastError(), "piovUpdate");
}


// wedge updateCoeffs (OF wedgeFvPatchField<vector>::evaluate). The mixed slot already holds the
// valueFraction d_k = 0.5*(1 - cellT_kk), which is pure geometry; what changes each step is the value,
//     target_k = (faceT & U_cell)_k
// and the mixed blend d*ref + (1-d)*U_c reproduces it with
//     ref_k = (target_k - (1 - d_k)*U_c[k]) / d_k .
// d_k is exactly zero on the AXIS component -- the rotation leaves it alone -- and there the blend is
// already pure zeroGradient, which equals target_k, so ref_k is multiplied by zero and left at zero
// rather than divided by it.
__device__
void wedgeUpdateKernel(
    int n,
    const label* __restrict__ wdg,
    const label* __restrict__ fc,
    const scalar* __restrict__ T,     // 9*n, row-major faceT per face
    const scalar* __restrict__ vf0, const scalar* __restrict__ vf1, const scalar* __restrict__ vf2,
    const scalar* __restrict__ Ux, const scalar* __restrict__ Uy, const scalar* __restrict__ Uz,
    scalar* __restrict__ r0, scalar* __restrict__ r1, scalar* __restrict__ r2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !wdg[i]) return;

    const int c = fc[i];
    const scalar u[3] = { Ux[c], Uy[c], Uz[c] };
    const scalar d[3] = { vf0[i], vf1[i], vf2[i] };
    scalar* r[3] = { r0 + i, r1 + i, r2 + i };
    for (int k = 0; k < 3; ++k)
    {
        const scalar tgt = T[9*i + 3*k + 0]*u[0] + T[9*i + 3*k + 1]*u[1] + T[9*i + 3*k + 2]*u[2];
        // ref = u + (tgt - u)/d, NOT (tgt - (1 - d)u)/d. They are the same algebraically and not at all
        // the same in floating point: d = 0.5*(1 - cos 2th) is ~1.9e-3 for OpenFOAM's recommended 2.5 deg
        // wedge, so the second form divides the difference of two nearly equal numbers by 2e-3 and
        // amplifies the cancellation ~500x. The rotation increment (tgt - u) is small and exact, so this
        // form carries no cancellation at all. Measured on movingCone: the spurious flux through the
        // wedge planes fell from 3.7e-10 to OpenFOAM's own 1e-11 level.
        *r[k] = (d[k] > scalar(1e-30)) ? (u[k] + (tgt - u[k]) / d[k]) : u[k];
    }
}

void deviceUpdateWedge(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux,
                       const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz)
{
    const int n = dbU.n;
    if (n == 0 || dbU.comp[0].wedgeMask.size() == 0) return;
    {
        const label* wdg = dbU.comp[0].wedgeMask.data(); const label* fc = dbU.comp[0].faceCell.data();
        const scalar* T = dbU.comp[0].wedgeT.data();
        const scalar* vf0 = dbU.comp[0].valueFraction.data(); const scalar* vf1 = dbU.comp[1].valueFraction.data();
        const scalar* vf2 = dbU.comp[2].valueFraction.data();
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        scalar* r0 = dbU.comp[0].refValue.data(); scalar* r1 = dbU.comp[1].refValue.data(); scalar* r2 = dbU.comp[2].refValue.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            wedgeUpdateKernel(n, wdg, fc, T, vf0, vf1, vf2, Uxd, Uyd, Uzd, r0, r1, r2); });
    }
    cudaCheck(cudaGetLastError(), "updateWedge");
}


namespace {
// The wedge value of ANY vector field is that field's own rotated cell value -- not a blend against a
// refValue derived from some other field. OF gets this for free: fvMatrix::H() is constructed with psi's
// BC types, so H carries wedge patches and H.correctBoundaryConditions() rotates H itself; rAU's wedge is
// the SCALAR wedge (= zeroGradient), so HbyA_b = rAU_c*(faceT & H_c) = faceT & HbyA_c, and its flux
// through the wedge plane is then identically zero. Evaluating HbyA through U's refValue instead leaves a
// residual (faceT & U_c) - (faceT & HbyA_c) on every wedge face, which on movingCone's 3800 of them was a
// net leak the pressure equation answered with a fictitious inflow at the open end.
__device__
void wedgeFaceValueKernel(
    int n,
    const label* __restrict__ wdg,
    const label* __restrict__ fc,
    const scalar* __restrict__ T,     // 9*n, row-major faceT per face
    const scalar* __restrict__ fx, const scalar* __restrict__ fy, const scalar* __restrict__ fz,
    scalar* __restrict__ bx, scalar* __restrict__ by, scalar* __restrict__ bz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !wdg[i]) return;

    const int c = fc[i];
    const scalar v[3] = { fx[c], fy[c], fz[c] };
    scalar* b[3] = { bx + i, by + i, bz + i };
    for (int k = 0; k < 3; ++k)
        *b[k] = T[9*i + 3*k + 0]*v[0] + T[9*i + 3*k + 1]*v[1] + T[9*i + 3*k + 2]*v[2];
}
} // namespace

void deviceWedgeFaceValue(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& fx,
    const DeviceBuffer<scalar>& fy,
    const DeviceBuffer<scalar>& fz,
    DeviceBuffer<scalar>& bx,
    DeviceBuffer<scalar>& by,
    DeviceBuffer<scalar>& bz)
{
    const int n = dbU.n;
    if (n == 0 || dbU.comp[0].wedgeMask.size() == 0) return;
    {
        const label* wdg = dbU.comp[0].wedgeMask.data(); const label* fc = dbU.comp[0].faceCell.data();
        const scalar* T = dbU.comp[0].wedgeT.data();
        const scalar* fxd = fx.data(); const scalar* fyd = fy.data(); const scalar* fzd = fz.data();
        scalar* bxd = bx.data(); scalar* byd = by.data(); scalar* bzd = bz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            wedgeFaceValueKernel(n, wdg, fc, T, fxd, fyd, fzd, bxd, byd, bzd); });
    }
    cudaCheck(cudaGetLastError(), "wedgeFaceValue");
}


namespace {
// OF correctUphiBCs: phi_b = U_b & Sf on every patch whose U field FIXES its value. `adjustable` is the
// adjustPhi mask, which is already !fixesValue() over the same face order -- so the faces to overwrite
// are exactly the zeros. Kept as its own kernel rather than folded into the flux computation: the
// unmasked faces must keep the flux the Sf&Uf remap just gave them.
__device__
void selectFixedFluxKernel(int n, const label* __restrict__ adjustable,
                           const scalar* __restrict__ phiFixed, scalar* __restrict__ phiB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || adjustable[i]) return;
    phiB[i] = phiFixed[i];
}
} // namespace

void deviceSelectFixedFlux(const DeviceBuffer<label>& adjustable,
                           const DeviceBuffer<scalar>& phiFixed,
                           DeviceBuffer<scalar>& phiB)
{
    const int n = static_cast<int>(phiB.size());
    if (n == 0 || static_cast<int>(adjustable.size()) != n || static_cast<int>(phiFixed.size()) != n) return;
    {
        const label* adjustableD = adjustable.data(); const scalar* phiFixedD = phiFixed.data(); scalar* phiBd = phiB.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            selectFixedFluxKernel(n, adjustableD, phiFixedD, phiBd); });
    }
    cudaCheck(cudaGetLastError(), "selectFixedFlux");
}


void deviceUpdateSymmetry(
    DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz)
{
    const int n = dbU.n;
    if (n == 0) return;
    {
        const label* sym = dbU.comp[0].symMask.data(); const label* fc = dbU.comp[0].faceCell.data();
        const scalar* nxd = dbU.nx.data(); const scalar* nyd = dbU.ny.data(); const scalar* nzd = dbU.nz.data();
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        scalar* vf0 = dbU.comp[0].valueFraction.data(); scalar* vf1 = dbU.comp[1].valueFraction.data(); scalar* vf2 = dbU.comp[2].valueFraction.data();
        scalar* r0 = dbU.comp[0].refValue.data(); scalar* r1 = dbU.comp[1].refValue.data(); scalar* r2 = dbU.comp[2].refValue.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            symUpdateKernel(n, sym, fc, nxd, nyd, nzd, Uxd, Uyd, Uzd, vf0, vf1, vf2, r0, r1, r2); });
    }
    cudaCheck(cudaGetLastError(), "symUpdate");
}


void deviceConstrainSymmetryHbyA(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Hx,
    const DeviceBuffer<scalar>& Hy,
    const DeviceBuffer<scalar>& Hz,
    DeviceBuffer<scalar>& hbx,
    DeviceBuffer<scalar>& hby,
    DeviceBuffer<scalar>& hbz)
{
    const int n = dbU.n;
    if (n == 0) return;
    {
        const label* sym = dbU.comp[0].symMask.data(); const label* fc = dbU.comp[0].faceCell.data();
        const scalar* nxd = dbU.nx.data(); const scalar* nyd = dbU.ny.data(); const scalar* nzd = dbU.nz.data();
        const scalar* Hxd = Hx.data(); const scalar* Hyd = Hy.data(); const scalar* Hzd = Hz.data();
        scalar* hbxd = hbx.data(); scalar* hbyd = hby.data(); scalar* hbzd = hbz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            symHbyAKernel(n, sym, fc, nxd, nyd, nzd, Hxd, Hyd, Hzd, hbxd, hbyd, hbzd); });
    }
    cudaCheck(cudaGetLastError(), "constrainSymmetryHbyA");
}


void deviceExtrapolateIOHbyA(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& extx,
    const DeviceBuffer<scalar>& exty,
    const DeviceBuffer<scalar>& extz,
    DeviceBuffer<scalar>& hbx,
    DeviceBuffer<scalar>& hby,
    DeviceBuffer<scalar>& hbz)
{
    const int n = dbU.n;
    if (n == 0) return;
    {
        const label* ioMask = dbU.comp[0].ioMask.data();
        const scalar* extxd = extx.data(); const scalar* extyd = exty.data(); const scalar* extzd = extz.data();
        scalar* hbxd = hbx.data(); scalar* hbyd = hby.data(); scalar* hbzd = hbz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            selectIOKernel(n, ioMask, extxd, extyd, extzd, hbxd, hbyd, hbzd); });
    }
    cudaCheck(cudaGetLastError(), "extrapolateIOHbyA");
}


void deviceConstrainMixedHbyA(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& hbx,
    DeviceBuffer<scalar>& hby,
    DeviceBuffer<scalar>& hbz)
{
    const int n = dbU.n;
    if (n == 0) return;
    DeviceBuffer<scalar> ubx, uby, ubz;   // U_b = mixed boundary value of U (uses dbU vf); deviceBCValue in device_boundary_assembly.cu
    deviceBCValue(dbU.comp[0], Ux, ubx);
    deviceBCValue(dbU.comp[1], Uy, uby);
    deviceBCValue(dbU.comp[2], Uz, ubz);
    {
        const label* mask = dbU.comp[0].mixedMask.data();
        const scalar* ubxd = ubx.data(); const scalar* ubyd = uby.data(); const scalar* ubzd = ubz.data();
        scalar* hbxd = hbx.data(); scalar* hbyd = hby.data(); scalar* hbzd = hbz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            selectMixedKernel(n, mask, ubxd, ubyd, ubzd, hbxd, hbyd, hbzd); });
    }
    cudaCheck(cudaGetLastError(), "constrainMixedHbyA");
}


namespace {
// patchInternalField for EVERY boundary face: out[i] = cellField[faceCell[i]]. OF's
// fvPatchField::patchInternalField(). Small (nBndFaces), so a caller can pull it to the host and do a
// patch-wide reduction there without moving the whole cell field.
__device__
void gatherPatchInternalKernel(int n, const label* __restrict__ faceCell,
                               const scalar* __restrict__ cellField, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cellField[faceCell[i]];
}
}   // namespace

void deviceGatherPatchInternal(const DeviceBoundary& db, const DeviceBuffer<scalar>& cellField,
                               DeviceBuffer<scalar>& out)
{
    out.resize(db.n);
    if (db.n == 0) return;
    {
        const int n = db.n;
        const label* fc = db.faceCell.data(); const scalar* cellFieldD = cellField.data(); scalar* outd = out.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            gatherPatchInternalKernel(n, fc, cellFieldD, outd); });
    }
    cudaCheck(cudaGetLastError(), "gatherPatchInternal");
}


void deviceUpdateTotalPressure(
    DeviceBoundary& db,
    const DeviceBuffer<scalar>& phiB,
    const DeviceBuffer<scalar>& Uxb,
    const DeviceBuffer<scalar>& Uyb,
    const DeviceBuffer<scalar>& Uzb,
    const DeviceBuffer<scalar>* rhoBnd)
{
    if (db.n == 0) return;
    {
        const int n = db.n;
        const label* tpMask = db.tpMask.data(); const scalar* p0 = db.p0.data(); const scalar* phiBd = phiB.data();
        const scalar* Uxbd = Uxb.data(); const scalar* Uybd = Uyb.data(); const scalar* Uzbd = Uzb.data();
        const scalar* rhoBndD = (rhoBnd && rhoBnd->size() == static_cast<std::size_t>(n)) ? rhoBnd->data() : nullptr;
        scalar* refValueD = db.refValue.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            tpUpdateKernel(n, tpMask, p0, phiBd, Uxbd, Uybd, Uzbd, rhoBndD, refValueD); });
    }
    cudaCheck(cudaGetLastError(), "tpUpdate");
}

} // namespace brae

namespace brae {
namespace {

// turbulentIntensityKineticEnergyInlet: refValue = 1.5*I^2*|Up|^2, from the CURRENT boundary U.
__device__
void tkeInletKernel(
    int n,
    const label* __restrict__ mask,
    const scalar* __restrict__ intensity,
    const scalar* __restrict__ ux,
    const scalar* __restrict__ uy,
    const scalar* __restrict__ uz,
    scalar* __restrict__ kRef)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !mask[i]) return;
    const scalar u2 = ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i];
    kRef[i] = scalar(1.5) * intensity[i] * intensity[i] * u2;
}

// turbulentMixingLength{DissipationRate,Frequency}Inlet, from the CURRENT boundary k.
//   epsilon: (Cmu^0.75/L)*k^1.5      omega: sqrt(k)/(Cmu^0.25*L)
__device__
void mixingLengthInletKernel(
    int n,
    const label* __restrict__ mask,      // 1 = epsilon, 2 = omega
    const scalar* __restrict__ len,
    const scalar* __restrict__ kRef,
    scalar Cmu75,
    scalar Cmu25,
    scalar* __restrict__ sRef)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !mask[i]) return;
    const scalar kb = fmax(kRef[i], scalar(0));
    sRef[i] = (mask[i] == 1) ? Cmu75 * pow(kb, scalar(1.5)) / len[i]
                             : sqrt(kb) / (Cmu25 * len[i]);
}

}   // namespace

// OF re-evaluates both of these in updateCoeffs EVERY outer iteration, so they track the solution. brae
// evaluated them once on the host at set-up. That is exact for a fixedValue U inlet (Up never moves) and
// wrong for flowRateInletVelocity, where Up is rebuilt each iteration from the live boundary density: the
// set-up value uses the SEED density (rhoInlet, or 1.0 when absent), so on angledDuct the frozen inlet |U|
// is ~1.19x too large and k_inlet lands ~41% high, epsilon ~67% high.
void deviceUpdateTurbulentInletK(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<label>& mask,
    const DeviceBuffer<scalar>& intensity,
    DeviceBoundary& dbK)
{
    const int n = dbK.n;
    if (!n || !mask.size()) return;
    {
        const label* maskD = mask.data(); const scalar* intensityD = intensity.data();
        const scalar* r0 = dbU.comp[0].refValue.data(); const scalar* r1 = dbU.comp[1].refValue.data();
        const scalar* r2 = dbU.comp[2].refValue.data(); scalar* kRefD = dbK.refValue.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            tkeInletKernel(n, maskD, intensityD, r0, r1, r2, kRefD); });
    }
    cudaCheck(cudaGetLastError(), "tkeInlet");
}

void deviceUpdateTurbulentInletSecond(
    const DeviceBoundary& dbK,
    const DeviceBuffer<label>& mask,
    const DeviceBuffer<scalar>& len,
    scalar Cmu,
    DeviceBoundary& dbSecond)
{
    const int n = dbSecond.n;
    if (!n || !mask.size()) return;
    {
        const label* maskD = mask.data(); const scalar* lenD = len.data(); const scalar* kRefD = dbK.refValue.data();
        const scalar cmu75 = pow(Cmu, scalar(0.75)); const scalar cmu25 = pow(Cmu, scalar(0.25));
        scalar* sRefD = dbSecond.refValue.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            mixingLengthInletKernel(n, maskD, lenD, kRefD, cmu75, cmu25, sRefD); });
    }
    cudaCheck(cudaGetLastError(), "mixingLengthInlet");
}

}   // namespace brae
