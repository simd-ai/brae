// rhoCentralStep.cu -- the device arm. See the header; the arithmetic is the host arm's (rhoCentralStep_cpp.cu).
//
// Per step:
//   centralFluxes (rhoCentralFlux.cu)                 rhoCentralFoam.C:102-210
//   cellKernel   fvc::div gathered in OpenFOAM's face order, then the four cell stages   :222-286
//   deviceUpdateInletOutlet on U, T, p                 :195's phi decides each inletOutlet face (shared, milestone D)
//   deviceBCValue on U (x3)                            :232 U.correctBoundaryConditions (shared DeviceBoundary)
//   deviceBCValue on T with the cells' old T           :265 Tw.evaluate() inside the energy BCs
//   patchKernel  transformPatchU, updatePatchFace      :232 on the transform faces, :233, 265-271
//   deviceBCValue on p                                 :287
//   rhoBKernel   rho_b = psi_b*p_b                     :288
//   courantKernel                                       centralCourantNo.H, kept on the device
#include "rhoCentralStep.cuh"
#include "rhoCentralCellUpdate.cuh"
#include "rhoCentralFlux.cuh"
#include "rhoCentralPatchUpdate.cuh"
#include "rhoCentralViscous.cuh"
#include "rhoCentralSlipBCs.cuh"

#include "device_boundary.cuh"
#include "device_fvc_smooth.cuh"      // fvc::smooth for LTS (milestone I)
#include "fvc_smooth_cpp.cuh"         // ...and the host wave, its oracle
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"   // mesh.update() (milestone K)
#include "face_cpp.cuh"                // face::centre (movingWallVelocity)
#include "primitive_patch_cpp.cuh"    // CellFaces
#include "inter_phase_time.cuh"        // BRAE_INTER_PHASE_TIME (off by default)

#include <cuda_runtime.h>
#include <optional>
#include <string>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

namespace {

constexpr int kTPB = 256;
inline int blocksFor(int n) { return (n + kTPB - 1)/kTPB; }
constexpr int kReportWindow = 256;

struct CellState
{
    scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rhoE, *e, *T, *Told, *p, *psi, *Ux, *Uy, *Uz;
};
struct FluxesIn
{
    const scalar *phi, *phiUx, *phiUy, *phiUz, *phiEp, *amaxSf;
    const scalar *phiB, *phiUxB, *phiUyB, *phiUzB, *phiEpB, *amaxSfB;
};

// fvc::div (surfaceIntegrate, fvcSurfaceIntegrate.C:45-78) gathered in OpenFOAM's scatter order -- owner (+) and
// neighbour (-) faces merged by face index, then the non-empty patch faces (sums only: no product to round)
struct DivGather
{
    scalar d0, d1, d2, d3, d4, sa;
};
__device__ __forceinline__ DivGather gatherDiv(int c, const label* __restrict__ ownerStart,
                                               const label* __restrict__ losort, const label* __restrict__ losortStart,
                                               const label* __restrict__ bndCellStart, const label* __restrict__ bndPerm,
                                               const label* __restrict__ bndIsEmpty, const FluxesIn& Fx)
{
    scalar d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, sa = 0;
    int fo = ownerStart[c];
    const int foEnd = ownerStart[c + 1];
    int kn = losortStart[c];
    const int knEnd = losortStart[c + 1];
    while (fo < foEnd || kn < knEnd)
    {
        if ((kn >= knEnd) || (fo < foEnd && fo < losort[kn]))
        {
            d0 += Fx.phi[fo];
            d1 += Fx.phiUx[fo];
            d2 += Fx.phiUy[fo];
            d3 += Fx.phiUz[fo];
            d4 += Fx.phiEp[fo];
            sa += Fx.amaxSf[fo];
            ++fo;
        }
        else
        {
            const int f = losort[kn];
            d0 -= Fx.phi[f];
            d1 -= Fx.phiUx[f];
            d2 -= Fx.phiUy[f];
            d3 -= Fx.phiUz[f];
            d4 -= Fx.phiEp[f];
            sa += Fx.amaxSf[f];
            ++kn;
        }
    }
    for (int q = bndCellStart[c]; q < bndCellStart[c + 1]; ++q)
    {
        const int k = bndPerm[q];
        if (bndIsEmpty[k]) continue;
        d0 += Fx.phiB[k];
        d1 += Fx.phiUxB[k];
        d2 += Fx.phiUyB[k];
        d3 += Fx.phiUzB[k];
        d4 += Fx.phiEpB[k];
        sa += Fx.amaxSfB[k];
    }
    return DivGather{d0, d1, d2, d3, d4, sa};
}

// fvc::div (surfaceIntegrate, fvcSurfaceIntegrate.C:45-78) gathered in OpenFOAM's scatter order -- owner (+) and
// neighbour (-) faces merged by face index, then the non-empty patch faces, then / V -- and the cell's update.
// surfaceSum(amaxSf) for the Courant number rides along (+ on both sides, fvcSurfaceIntegrate.C:165-166).
// kMoving (milestone K): a moving mesh's instantiation, its stages on V0; the static one is the kernel as it was, so a
// static mesh's arithmetic is the same to the bit
template <bool kMoving>
__global__ void cellKernel(int nC, int slot, scalar rDeltaT0, int v0, int v1, int v2, GasCoeffs gas,
                           const label* __restrict__ ownerStart, const label* __restrict__ losort,
                           const label* __restrict__ losortStart, const label* __restrict__ bndCellStart,
                           const label* __restrict__ bndPerm, const label* __restrict__ bndIsEmpty,
                           const scalar* __restrict__ V, FluxesIn Fx, CellState S,
                           scalar* __restrict__ coCell, int* __restrict__ status, const scalar* __restrict__ rDT,
                           const scalar* __restrict__ V0)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar rDeltaT = rDT ? rDT[c] : rDeltaT0;   // localEuler: the cell's own (milestone I)
    const DivGather g = gatherDiv(c, ownerStart, losort, losortStart, bndCellStart, bndPerm, bndIsEmpty, Fx);
    scalar d0 = g.d0, d1 = g.d1, d2 = g.d2, d3 = g.d3, d4 = g.d4;
    const scalar sa = g.sa;
    const scalar Vc = V[c];
    coCell[c] = sa;
    d0 /= Vc;
    d1 /= Vc;
    d2 /= Vc;
    d3 /= Vc;
    d4 /= Vc;

    // the four stages of rhoCentralCellUpdate.cuh, back to back
    const scalar Told = S.T[c];
    // the moving mesh's old volume (milestone K) in its own stage functions: a static mesh keeps the stages it had,
    // expression for expression (one shared V0c = Vc changed nvcc's multiply-add contraction there)
    MomentumNew m;
    EnergyNew en;
    if constexpr (kMoving)
    {
        m = momentumStageMoving(rDeltaT, Vc, V0[c], d0, d1, d2, d3, v0, v1, v2, S.rho[c], S.rhoUx[c], S.rhoUy[c],
                                S.rhoUz[c]);
        en = energyStageMoving(rDeltaT, Vc, V0[c], d4, S.rhoE[c], m.rho, m.Ux, m.Uy, m.Uz);
    }
    else
    {
        (void)V0;
        m = momentumStage(rDeltaT, Vc, d0, d1, d2, d3, v0, v1, v2, S.rho[c], S.rhoUx[c], S.rhoUy[c], S.rhoUz[c]);
        en = energyStage(rDeltaT, Vc, d4, S.rhoE[c], m.rho, m.Ux, m.Uy, m.Uz);
    }
    const ThermoNew th = thermoStage(en.e, S.p[c], Told, gas);
    const scalar p = pressureStage(m.rho, th.psi);
    S.rho[c] = m.rho;
    S.rhoUx[c] = m.rhoUx;
    S.rhoUy[c] = m.rhoUy;
    S.rhoUz[c] = m.rhoUz;
    S.Ux[c] = m.Ux;
    S.Uy[c] = m.Uy;
    S.Uz[c] = m.Uz;
    S.rhoE[c] = en.rhoE;
    S.e[c] = en.e;
    S.Told[c] = Told;
    S.T[c] = th.T;
    S.psi[c] = th.psi;
    S.p[c] = p;
    const int st = th.status | physicalStatus(m.rho, p, th.T, en.e);
    if (st) atomicOr(&status[slot], st);
}

struct PatchState
{
    scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rhoE, *e, *T, *psi, *Ux, *Uy, *Uz;
    const scalar *p, *TbEval;
};
// the cells' new rho and U, and each face's transform (rhoCentralPatchUpdate.cuh; Case::bUTransform and friends)
struct PatchTransforms
{
    const scalar *rhoC, *UxC, *UyC, *UzC;
    const int* kind;
    const tensor* R;
    const char *rhoConstraint, *TseedOwn;
};

// ioCount: the inletOutlet diagnostic, over the faces where U, p or T is inletOutlet -- [0] the face-steps whose phi was
// negative (inflow), [1] all of them
__global__ void patchKernel(int nB, int slot, GasCoeffs gas, int ctrl, const label* __restrict__ bCell,
                            const int* __restrict__ TfixesValue, const scalar* __restrict__ eCell, PatchState B,
                            PatchTransforms X, int* __restrict__ status, const scalar* __restrict__ phiB,
                            const char* __restrict__ ioFace, unsigned long long* __restrict__ ioCount)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    const label c = bCell[k];
    if (ioCount && ioFace[k])   // counted once a step: the viscous step updates the patches twice (:265-271, :280)
    {
        atomicAdd(&ioCount[1], 1ull);
        if (phiB[k] < 0) atomicAdd(&ioCount[0], 1ull);
    }
    if (X.kind[k])   // U.correctBoundaryConditions() on a transform face (:232)
    {
        const vector Ub = transformPatchU(X.kind[k], X.R[k], vector{X.UxC[c], X.UyC[c], X.UzC[c]}, ctrl);
        B.Ux[k] = Ub.x;
        B.Uy[k] = Ub.y;
        B.Uz[k] = Ub.z;
    }
    const PatchNew b = updatePatchFace(TfixesValue[k], patchRhoMid(X.rhoConstraint[k], X.rhoC[c], B.rho[k], ctrl),
                                       B.Ux[k], B.Uy[k], B.Uz[k], eCell[c], X.TseedOwn[k] ? B.T[k] : B.TbEval[k],
                                       B.p[k], gas);
    B.rhoUx[k] = b.rhoUx;
    B.rhoUy[k] = b.rhoUy;
    B.rhoUz[k] = b.rhoUz;
    B.e[k] = b.e;
    B.T[k] = b.T;
    B.psi[k] = b.psi;
    B.rhoE[k] = b.rhoE;
    if (b.status) atomicOr(&status[slot], b.status);
}

// fixedEnergy::updateCoeffs on a totalTemperature face at the e matrix's construction (milestone H): Tw.evaluate() has
// just re-evaluated T_b (refValue), and he_b = thermo.he(p_b, T_b). Nothing else moves: T_b's own slot, psi_b and alpha_b
// keep the first thermo.correct()'s until the correct after the solve.
__global__ void fixedEnergyFaceKernel(int n, const label* __restrict__ faces, GasCoeffs gas,
                                      const scalar* __restrict__ Tref, const scalar* __restrict__ pB,
                                      scalar* __restrict__ eB)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label k = faces[i];
    eB[k] = gasEs(pB[k], Tref[k], gas);
}

// the control inletOutletInverted: the switch fed -phi
__global__ void negateKernel(int n, const scalar* __restrict__ a, scalar* __restrict__ b)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < n) b[k] = -a[k];
}

// rho.boundaryFieldRef() == psi.boundaryField()*p.boundaryField(), rhoCentralFoam.C:288
__global__ void rhoBKernel(int nB, const scalar* __restrict__ psiB, const scalar* __restrict__ pB,
                           scalar* __restrict__ rhoB)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    rhoB[k] = psiB[k]*pB[k];
}

// centralCourantNo.H: CoNum = 0.5*gMax(sumAmaxSf/V)*dt, meanCoNum = 0.5*(gSum(sumAmaxSf)/gSum(V))*dt
// ---- local time stepping (milestone I), setRDeltaT.H -------------------------------------------------------------
// fvc::surfaceSum(amaxSf), the gather the cell kernels make too (its sa), ahead of them: rDeltaT needs it first
__global__ void surfaceSumKernel(int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
                                 const label* __restrict__ losortStart, const label* __restrict__ bndCellStart,
                                 const label* __restrict__ bndPerm, const label* __restrict__ bndIsEmpty, FluxesIn Fx,
                                 scalar* __restrict__ sa)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    sa[c] = gatherDiv(c, ownerStart, losort, losortStart, bndCellStart, bndPerm, bndIsEmpty, Fx).sa;
}
// setRDeltaT.H:14-19: max(1/maxDeltaT, surfaceSum(amaxSf)/((2*maxCo)*V)) -- (2*maxCo)*V is a field first, then the
// quotient; max(s, x) is (s > x) ? s : x
__global__ void rDeltaTKernel(int nC, scalar rMaxDeltaT, scalar twoMaxCo, const scalar* __restrict__ sa,
                              const scalar* __restrict__ V, scalar* __restrict__ rDT)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar x = sa[c]/__dmul_rn(twoMaxCo, V[c]);
    rDT[c] = (rMaxDeltaT > x) ? rMaxDeltaT : x;
}

__global__ void courantKernel(int nC, const scalar* __restrict__ sumA, const scalar* __restrict__ V,
                              double* __restrict__ out)
{
    __shared__ double smax[1024], ssum[1024], sv[1024];
    const int t = threadIdx.x;
    double mx = -1e300, sa = 0, svol = 0;
    for (int i = t; i < nC; i += blockDim.x)
    {
        const double r = sumA[i]/V[i];
        mx = (r > mx) ? r : mx;
        sa += sumA[i];
        svol += V[i];
    }
    smax[t] = mx;
    ssum[t] = sa;
    sv[t] = svol;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            smax[t] = (smax[t + s] > smax[t]) ? smax[t + s] : smax[t];
            ssum[t] += ssum[t + s];
            sv[t] += sv[t + s];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        out[0] = smax[0];
        out[1] = ssum[0];
        out[2] = sv[0];
    }
}

// ---- the viscous step's cell kernels (milestone G): the fused cellKernel's stages, split where the correctors go -----
// totalPressure's rho form (milestone K, movingCone's `left`: psi none), totalPressureFvPatchScalarField.C:
// p0 - 0.5*rho_b*neg(phi_b)*magSqr(U_b), each field product rounded alone; the psi form's faces are left as they are
__global__ void totalPressureRhoKernel(int n, const label* __restrict__ faces, const char* __restrict__ rhoForm,
                                       const scalar* __restrict__ p0, const scalar* __restrict__ phi,
                                       const scalar* __restrict__ ux, const scalar* __restrict__ uy,
                                       const scalar* __restrict__ uz, const scalar* __restrict__ rho,
                                       scalar* __restrict__ ref)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n || !rhoForm[i]) return;
    const label k = faces[i];
    const scalar ng = (phi[k] < 0) ? scalar(1) : scalar(0);   // Foam::neg
    const scalar m = ux[k]*ux[k] + uy[k]*uy[k] + uz[k]*uz[k];   // magSqr
    const scalar t = __dmul_rn(__dmul_rn(__dmul_rn(scalar(0.5), rho[k]), ng), m);
    ref[k] = p0[i] - t;
}

// :222-232, the explicit rho and rhoU and U = rhoU/rho; keeps div(phiEp) (over V) for the energy stage and T as the
// step starts (Tw.evaluate()'s and the e -> T inversion's seed)
__global__ void momentumKernel(int nC, scalar rDeltaT0, int v0, int v1, int v2, const label* __restrict__ ownerStart,
                               const label* __restrict__ losort, const label* __restrict__ losortStart,
                               const label* __restrict__ bndCellStart, const label* __restrict__ bndPerm,
                               const label* __restrict__ bndIsEmpty, const scalar* __restrict__ V, FluxesIn Fx,
                               CellState S, scalar* __restrict__ coCell, scalar* __restrict__ divPhiEp,
                               const scalar* __restrict__ rDT, const scalar* __restrict__ V0)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar rDeltaT = rDT ? rDT[c] : rDeltaT0;
    const DivGather g = gatherDiv(c, ownerStart, losort, losortStart, bndCellStart, bndPerm, bndIsEmpty, Fx);
    scalar d0 = g.d0, d1 = g.d1, d2 = g.d2, d3 = g.d3, d4 = g.d4;
    const scalar Vc = V[c];
    coCell[c] = g.sa;
    d0 /= Vc;
    d1 /= Vc;
    d2 /= Vc;
    d3 /= Vc;
    d4 /= Vc;
    divPhiEp[c] = d4;
    S.Told[c] = S.T[c];
    const MomentumNew m = V0 ? momentumStageMoving(rDeltaT, Vc, V0[c], d0, d1, d2, d3, v0, v1, v2, S.rho[c], S.rhoUx[c],
                                                   S.rhoUy[c], S.rhoUz[c])
                             : momentumStage(rDeltaT, Vc, d0, d1, d2, d3, v0, v1, v2, S.rho[c], S.rhoUx[c], S.rhoUy[c],
                                             S.rhoUz[c]);
    S.rho[c] = m.rho;
    S.rhoUx[c] = m.rhoUx;
    S.rhoUy[c] = m.rhoUy;
    S.rhoUz[c] = m.rhoUz;
    S.Ux[c] = m.Ux;
    S.Uy[c] = m.Uy;
    S.Uz[c] = m.Uz;
}
// :258-266, rhoE explicit with - fvc::div(sigmaDotU) (dS; null under the control noSigmaDotU: the inviscid stage),
// e = rhoE/rho - 0.5*magSqr(U) on the corrected U, thermo.correct()'s cell half on the old p
__global__ void energyKernel(int nC, int slot, scalar rDeltaT0, GasCoeffs gas, const scalar* __restrict__ V,
                             const scalar* __restrict__ divPhiEp, const scalar* __restrict__ dS, CellState S,
                             int* __restrict__ status, const scalar* __restrict__ rDT, const scalar* __restrict__ V0)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar rDeltaT = rDT ? rDT[c] : rDeltaT0;
    const scalar Vc = V[c];
    EnergyNew en;
    if (V0)   // a moving mesh (milestone K)
        en = dS ? energyStageViscousMoving(rDeltaT, Vc, V0[c], divPhiEp[c], dS[c], S.rhoE[c], S.rho[c], S.Ux[c], S.Uy[c],
                                           S.Uz[c])
                : energyStageMoving(rDeltaT, Vc, V0[c], divPhiEp[c], S.rhoE[c], S.rho[c], S.Ux[c], S.Uy[c], S.Uz[c]);
    else
        en = dS ? energyStageViscous(rDeltaT, Vc, divPhiEp[c], dS[c], S.rhoE[c], S.rho[c], S.Ux[c], S.Uy[c], S.Uz[c])
                : energyStage(rDeltaT, Vc, divPhiEp[c], S.rhoE[c], S.rho[c], S.Ux[c], S.Uy[c], S.Uz[c]);
    const ThermoNew th = thermoStage(en.e, S.p[c], S.Told[c], gas);
    S.rhoE[c] = en.rhoE;
    S.e[c] = en.e;
    S.T[c] = th.T;
    S.psi[c] = th.psi;
    if (th.status) atomicOr(&status[slot], th.status);
}
// :278-288 after the e corrector: thermo.correct()'s cell half (seeded with the T the first one left), rhoE =
// rho*(e + 0.5*magSqr(U)), p = rho/psi
__global__ void thermoPressureKernel(int nC, int slot, GasCoeffs gas, CellState S, int* __restrict__ status)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar e = S.e[c];
    const ThermoNew th = thermoStage(e, S.p[c], S.T[c], gas);
    const scalar rho = S.rho[c];
    const scalar Ux = S.Ux[c], Uy = S.Uy[c], Uz = S.Uz[c];
    S.rhoE[c] = rho*(e + 0.5*(Ux*Ux + Uy*Uy + Uz*Uz));
    S.T[c] = th.T;
    S.psi[c] = th.psi;
    const scalar p = pressureStage(rho, th.psi);
    S.p[c] = p;
    const int st = th.status | physicalStatus(rho, p, th.T, e);
    if (st) atomicOr(&status[slot], st);
}
// U.correctBoundaryConditions() on the transform faces (the rest is deviceBCValue's)
__global__ void uTransformKernel(int nB, int ctrl, const label* __restrict__ bCell, const int* __restrict__ kind,
                                 const tensor* __restrict__ R, const scalar* __restrict__ Ux,
                                 const scalar* __restrict__ Uy, const scalar* __restrict__ Uz,
                                 scalar* __restrict__ UxB, scalar* __restrict__ UyB, scalar* __restrict__ UzB)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB || !kind[k]) return;
    const label c = bCell[k];
    const vector Ub = transformPatchU(kind[k], R[k], vector{Ux[c], Uy[c], Uz[c]}, ctrl);
    UxB[k] = Ub.x;
    UyB[k] = Ub.y;
    UzB[k] = Ub.z;
}

void upCmpt(DeviceBuffer<scalar>& d, const std::vector<vector>& h, int j)
{
    std::vector<scalar> t(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) t[i] = (j == 0) ? h[i].x : (j == 1) ? h[i].y : h[i].z;
    d.copyFrom(t);
}

void downCmpt(const DeviceBuffer<scalar>& d, std::vector<vector>& h, int j)
{
    const std::vector<scalar> t = d.host();
    h.resize(t.size());
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        if (j == 0) h[i].x = t[i];
        else if (j == 1) h[i].y = t[i];
        else h[i].z = t[i];
    }
}


class DeviceArm final : public Arm
{
public:
    DeviceArm(const Case& cs, const State& s0, PatchFields& pf, int control) : cs_(cs), control_(control)
    {
        // the device memory the arm holds: what brae's device pool took from the driver to build it (every DeviceBuffer
        // goes through it). Not cudaMemGetInfo's free memory: on the GB10's unified memory that moves with every host
        // allocation: it read 0 and 303 MiB on wedge15Ma5 refined to 307,200 and 76,800 cells, which the pool puts at
        // 898.5 and 224.6 MiB.
        const std::size_t pool0 = brae::detail::devicePool().mallocBytes() - brae::detail::devicePool().freedBytes();
        const int nC = cs.nC, nB = cs.nB;
        dm_ = buildDeviceMesh(cs.m, cs.g, cs.patches);
        if (dm_.nBndFaces != cs.nB || dm_.nInternalFaces != cs.nIF)
            throw std::runtime_error("brae rhoCentralFoam: the device mesh's face layout is not the case's.");

        DeviceFields& F = F_;
        F.rho.copyFrom(s0.rho);
        upCmpt(F.rhoUx, s0.rhoU, 0);
        upCmpt(F.rhoUy, s0.rhoU, 1);
        upCmpt(F.rhoUz, s0.rhoU, 2);
        F.rhoE.copyFrom(s0.rhoE);
        F.e.copyFrom(s0.e);
        F.T.copyFrom(s0.T);
        F.p.copyFrom(s0.p);
        F.psi.copyFrom(s0.psi);
        upCmpt(F.Ux, s0.U, 0);
        upCmpt(F.Uy, s0.U, 1);
        upCmpt(F.Uz, s0.U, 2);
        F.rhoB.copyFrom(s0.rhoB);
        upCmpt(F.rhoUxB, s0.rhoUB, 0);
        upCmpt(F.rhoUyB, s0.rhoUB, 1);
        upCmpt(F.rhoUzB, s0.rhoUB, 2);
        F.rhoEB.copyFrom(s0.rhoEB);
        F.eB.copyFrom(s0.eB);
        F.TB.copyFrom(s0.TB);
        F.pB.copyFrom(s0.pB);
        F.psiB.copyFrom(s0.psiB);
        upCmpt(F.UxB, s0.UB, 0);
        upCmpt(F.UyB, s0.UB, 1);
        upCmpt(F.UzB, s0.UB, 2);
        X_.allocate(nC, cs.nIF, nB);

        // the patch fields, through brae's device boundary (the same flat layout: no coupled patch is accepted)
        dbU_ = buildDeviceVectorBoundary(pf.U, cs.patches, cs.g);
        // U's inletOutlet faces start with the switch OFF: OpenFOAM's dictionary constructor sets valueFraction 0
        // (inletOutletFvPatchField.C) and nothing updates U's patches before :218's fvc::grad(U), whose patch
        // correction takes the mixed snGrad, refGrad = 0 there. The shared builder seeds the face fixedValue(inletValue)
        // because its other callers switch before any read; here tauMC is the one read that comes first (milestone H,
        // measured on LadenburgJet60psi's outlet: 3.96e-06 m/s in the 20 outlet cells at step one, 0 with this).
        if (cs.ioU)
            for (int c = 0; c < 3; ++c)
            {
                std::vector<label> ty = dbU_.comp[c].bcType.host();
                const std::vector<label> io = dbU_.comp[c].ioMask.host();   // U's own inletOutlet faces
                for (label k = 0; k < nB; ++k)
                    if (io[k]) ty[k] = 0;
                dbU_.comp[c].bcType.copyFrom(ty);
            }
        dbP_ = buildDeviceBoundary(pf.p, cs.patches, cs.g);
        dbT_ = buildDeviceBoundary(pf.T, cs.patches, cs.g);
        if (dbP_.n != nB || dbT_.n != nB || dbU_.comp[0].n != nB)
            throw std::runtime_error("brae rhoCentralFoam: the device boundary's face layout is not the case's.");
        {
            std::vector<int> fx(nB);
            for (label k = 0; k < nB; ++k) fx[k] = pf.T.boundary[cs.bPatch[k]]->fixesValue() ? 1 : 0;
            TfixesValue_.copyFrom(fx);
        }
        if (cs.nozzleBoundaries())   // milestone H: the nozzle boundaries' faces, references and controls
        {
            tpFaces_.copyFrom(cs.tpFaces);
            tpP0_.copyFrom(cs.tpP0);
            tpGamma_.copyFrom(cs.tpGamma);
            tpRho_.copyFrom(cs.tpRho);
            for (char r : cs.tpRho) tpRhoAny_ = tpRhoAny_ || r;
            ttFaces_.copyFrom(cs.ttFaces);
            ttT0_.copyFrom(cs.ttT0);
            ttGamma_.copyFrom(cs.ttGamma);
            wtFaces_.copyFrom(cs.wtFaces);
            wtFieldInf_.copyFrom(cs.wtFieldInf);
            std::vector<scalar> li = cs.wtLInf;
            if (control_ == controlWaveNoRelax) for (scalar& v : li) v = -1;
            wtLInf_.copyFrom(li);
            wtGamma_.copyFrom(cs.wtGamma);
            if (!cs.wtFaces.empty())   // waveTransmissive is advective's MIXED face: bcType 5, refGrad 0
            {
                std::vector<label> ty = dbP_.bcType.host();
                for (label k : cs.wtFaces) ty[k] = 5;
                dbP_.bcType.copyFrom(ty);
                pBold_.resize(nB);
            }
            zeroB_.copyFrom(std::vector<scalar>(nB, 0.0));
            phiInv_.resize(nB);
        }
        bCell_.copyFrom(cs.bCell);
        pf_ = &pf;
        if (cs.motion)   // milestone K
        {
            // createFields.H:90, phi = fvc::flux(rhoU): what U.correctBoundaryConditions() switches on at the first
            // mesh.update(), before the first step's flux exists -- on the patches, rhoU_b & Sf_b
            std::vector<scalar> pb(nB, 0.0);
            for (label k = 0; k < nB; ++k)
            {
                if (cs.bEmpty[k]) continue;
                const vector S = cs.g.Sf()[static_cast<std::size_t>(cs.nIF + k)];
                const vector& rU = s0.rhoUB[static_cast<std::size_t>(k)];
                pb[k] = rU.x*S.x + rU.y*S.y + rU.z*S.z;
            }
            X_.phiB.copyFrom(pb);
            for (int c = 0; c < 3; ++c) refUhost_[c] = dbU_.comp[c].refValue.host();
        }
        uTransform_.copyFrom(cs.bUTransform);
        uR_.copyFrom(cs.bUR);
        rhoConstraint_.copyFrom(cs.bRhoConstraint);
        TseedOwn_.copyFrom(cs.bTseedOwn);
        ioFace_.copyFrom(cs.bInletOutlet);
        ioCount_.copyFrom(std::vector<unsigned long long>(2, 0ull));
        if (control_ == controlInletOutletInverted || control_ == controlLaggedPhi) phiCtrl_.resize(nB);
        Told_.resize(nC);
        TbEval_.resize(nB);
        coCell_.resize(nC);
        status_.copyFrom(std::vector<int>(kReportWindow, 0));
        courant_.copyFrom(std::vector<double>(3*kReportWindow, 0.0));
        // pinned, so flush() reads a window back with two async copies and one synchronisation: under adjustTimeStep
        // it runs every step
        cudaCheck(cudaMallocHost(&hostCourant_, 3*kReportWindow*sizeof(double)), "pinned courant");
        cudaCheck(cudaMallocHost(&hostStatus_, kReportWindow*sizeof(int)), "pinned status");
        outflow_.copyFrom(std::vector<double>(10, 0.0));   // the run's totals, then this step's sums
        if (cs.viscous)
        {
            vis_ = std::make_unique<ViscousCorrectors>(cs, dm_, control_);
            divPhiEp_.resize(nC);
            visIn_.copyFrom(std::vector<double>(4, 0.0));
        }
        // the rarefied-gas walls (milestone I): their descriptors, and partialSlip's coefficients from the written
        // refValue and valueFraction on the U the run starts from (the step's first fvc::grad(U) reads them)
        // RAS (milestone J): compressible::turbulenceModel::New(rho, U, phi, thermo), then turbulence->validate()
        // (rhoCentralFoam.C:71) -- correctNut() on the fields as read, with the thermo's mu as constructed
        if (pf.ras && vis_)
        {
            turb_ = std::make_unique<gpu::compressibleRAS::KOmegaSST>(*pf.ras, cs.m, cs.g, cs.patches, pf.U, dm_);
            vis_->setTurbulence(turb_.get());
            vis_->laminarMu(F_, muLamC_, muLamB_);
            gpu::compressibleRAS::Flow fv;   // correctNut reads no flux and no ddt (tools/default_audit_allow.txt)
            fv.rho = &F_.rho;
            fv.rhoB = &F_.rhoB;
            fv.Ux = &F_.Ux;
            fv.Uy = &F_.Uy;
            fv.Uz = &F_.Uz;
            fv.mu = &muLamC_;
            fv.muB = &muLamB_;
            turb_->validate(fv, dbU_);
            // BRAE_RAS_DUMP_VALIDATE=<file>: a diagnostic, off by default -- nut and alphat after validate(), cells
            // then patch faces, one value a line (compared against OpenFOAM's own validate(), which it never writes)
            if (const char* dump = std::getenv("BRAE_RAS_DUMP_VALIDATE"))
            {
                std::vector<scalar> a, b, c2, d;
                turb_->nut().copyTo(a);
                turb_->nutB().copyTo(b);
                turb_->alphat().copyTo(c2);
                turb_->alphatB().copyTo(d);
                if (std::FILE* fp = std::fopen(dump, "w"))
                {
                    for (const auto* v : {&a, &b, &c2, &d})
                    {
                        std::fprintf(fp, "%zu\n", v->size());
                        for (scalar x : *v) std::fprintf(fp, "%.17g\n", x);
                    }
                    std::fclose(fp);
                }
            }
        }
        if (cs.slipWalls())
        {
            slip_ = std::make_unique<SlipWalls>(cs, dm_, dbU_, dbT_, control_);
            slip_->emulate(dbU_, F_.Ux, F_.Uy, F_.Uz);
            if (slip_->smoluchowski()) vis_->setMixedEnergyRef(&slip_->eRef());
        }
        cudaCheck(cudaDeviceSynchronize(), "rhoCentralFoam upload");
        bytes_ = brae::detail::devicePool().mallocBytes() - brae::detail::devicePool().freedBytes() - pool0;
    }

    void step(scalar deltaT) override
    {
        if (pending_ >= kReportWindow)
            throw std::logic_error("brae rhoCentralFoam: device step with a full report window; flush() first.");
        if (vis_)
        {
            stepViscous(deltaT);
            return;
        }
        const Case& cs = cs_;
        DeviceFields& F = F_;
        const int nC = cs.nC, nB = cs.nB;
        keepOldPressureBoundary();

        centralFluxes(dm_, cs, control_, F, X_, outflow_.data(), deltaT, meshPhiI(), meshPhiB());

        const FluxesIn Fx{X_.phi.data(), X_.phiUx.data(), X_.phiUy.data(), X_.phiUz.data(), X_.phiEp.data(),
                          X_.amaxSf.data(), X_.phiB.data(), X_.phiUxB.data(), X_.phiUyB.data(), X_.phiUzB.data(),
                          X_.phiEpB.data(), X_.amaxSfB.data()};
        const CellState S{F.rho.data(), F.rhoUx.data(), F.rhoUy.data(), F.rhoUz.data(), F.rhoE.data(), F.e.data(),
                          F.T.data(), Told_.data(), F.p.data(), F.psi.data(), F.Ux.data(), F.Uy.data(), F.Uz.data()};
        const scalar* rDT = cs.lts ? setRDeltaT(Fx) : nullptr;   // :186-190, after centralCourantNo.H's amaxSf
        if (V0())   // milestone K
            cellKernel<true><<<blocksFor(nC), kTPB>>>(nC, pending_, 1.0/deltaT, cs.sd.valid(0), cs.sd.valid(1),
                                                cs.sd.valid(2), cs.gas, dm_.ownerStart.data(), dm_.losort.data(),
                                                dm_.losortStart.data(), dm_.bndCellStart.data(), dm_.bndPerm.data(),
                                                dm_.bndIsEmpty.data(), dm_.V.data(), Fx, S, coCell_.data(),
                                                status_.data(), rDT, V0());
        else
            cellKernel<false><<<blocksFor(nC), kTPB>>>(nC, pending_, 1.0/deltaT, cs.sd.valid(0), cs.sd.valid(1),
                                                cs.sd.valid(2), cs.gas, dm_.ownerStart.data(), dm_.losort.data(),
                                                dm_.losortStart.data(), dm_.bndCellStart.data(), dm_.bndPerm.data(),
                                                dm_.bndIsEmpty.data(), dm_.V.data(), Fx, S, coCell_.data(),
                                                status_.data(), rDT, V0());
        cudaCheck(cudaGetLastError(), "rhoCentral cells");

        if (nB)
        {
            switchInletOutlet();
            deviceBCValue(dbU_.comp[0], F.Ux, F.UxB);   // :232
            deviceBCValue(dbU_.comp[1], F.Uy, F.UyB);
            deviceBCValue(dbU_.comp[2], F.Uz, F.UzB);
            // :265, Tw.evaluate() on the cells' old T, before thermo.correct() (the control: on the new)
            updateTotalTemperature();
            deviceBCValue(dbT_, control_ == controlInletOutletNewT ? F.T : Told_, TbEval_);
            const PatchState B{F.rhoB.data(), F.rhoUxB.data(), F.rhoUyB.data(), F.rhoUzB.data(), F.rhoEB.data(),
                               F.eB.data(), F.TB.data(), F.psiB.data(), F.UxB.data(), F.UyB.data(), F.UzB.data(),
                               F.pB.data(), TbEval_.data()};
            const PatchTransforms X{F.rho.data(), F.Ux.data(), F.Uy.data(), F.Uz.data(), uTransform_.data(),
                                    uR_.data(), rhoConstraint_.data(), TseedOwn_.data()};
            patchKernel<<<blocksFor(nB), kTPB>>>(nB, pending_, cs.gas, control_, bCell_.data(), TfixesValue_.data(),
                                                 F.e.data(), B, X, status_.data(), X_.phiB.data(), ioFace_.data(),
                                                 ioCount_.data());
            cudaCheck(cudaGetLastError(), "rhoCentral patches");
            updatePressureBoundaries(deltaT);
            deviceBCValue(dbP_, F.p, F.pB);             // :287
            rhoBKernel<<<blocksFor(nB), kTPB>>>(nB, F.psiB.data(), F.pB.data(), F.rhoB.data());   // :288
            cudaCheck(cudaGetLastError(), "rhoCentral rho_b");
        }
        courantKernel<<<1, 1024>>>(nC, coCell_.data(), dm_.V.data(), courant_.data() + 3*pending_);
        cudaCheck(cudaGetLastError(), "rhoCentral courant");
        deltaT_[pending_] = deltaT;
        ++pending_;
    }

    // fvc::smooth(rDeltaT, coeff) (setRDeltaT.H): the host wave (cpu::fvc::smooth, the oracle) or the device wave
    // (deviceSmooth), the same cells to the bit (tests/interfoam_write/identity/wave_device.sh). Neither is faster on
    // every mesh -- on biconic25-55Run35 (32k cells) the host's is 7.8 ms and the device's 15.4 ms a call, on RAS/
    // DTCHull (846k) the host's 430 ms -- so the first call runs both, from the same field, refuses unless they agree
    // bit for bit, and keeps the faster. BRAE_RCF_SMOOTH=host or =device chooses one and skips the first call's check.
    void smoothRDeltaT()
    {
        const Case& cs = cs_;
        const scalar coeff = cs.rDeltaTSmoothingCoeff;
        if (smoothArm_ == SmoothArm::undecided)
        {
            const char* env = std::getenv("BRAE_RCF_SMOOTH");
            const std::string want = env ? env : "";
            if (want == "host") smoothArm_ = SmoothArm::host;
            else if (want == "device") smoothArm_ = SmoothArm::device;
            else
            {
                std::vector<scalar> d = rDThost_;
                const auto h0 = std::chrono::steady_clock::now();
                cpu::fvc::smooth(rDThost_, coeff, cs.m, cs.patches, &ltsCells_);
                const auto h1 = std::chrono::steady_clock::now();
                deviceSmooth(d, coeff, cs.m, cs.patches, ltsCells_, ltsWave_);
                const auto d1 = std::chrono::steady_clock::now();
                if (d.size() != rDThost_.size()
                 || std::memcmp(d.data(), rDThost_.data(), d.size()*sizeof(scalar)) != 0)
                    throw std::runtime_error("brae rhoCentralFoam: fvc::smooth's host and device waves gave different "
                                             "rDeltaT on the first step; they must agree to the bit (set "
                                             "BRAE_RCF_SMOOTH=host to run the oracle alone).");
                const double th = std::chrono::duration<double>(h1 - h0).count();
                const double td = std::chrono::duration<double>(d1 - h1).count();
                smoothArm_ = th <= td ? SmoothArm::host : SmoothArm::device;
                std::fprintf(stderr, "brae rhoCentralFoam: fvc::smooth: the host wave (%.3f ms) and the device wave "
                             "(%.3f ms) agree to the bit on the first call; the %s's is kept\n", 1e3*th, 1e3*td,
                             smoothArm_ == SmoothArm::host ? "host" : "device");
                return;
            }
        }
        if (smoothArm_ == SmoothArm::host)
        {
            interPhase::Nested t("fvc::smooth: the wave (host)");
            cpu::fvc::smooth(rDThost_, coeff, cs.m, cs.patches, &ltsCells_);
        }
        else
            deviceSmooth(rDThost_, coeff, cs.m, cs.patches, ltsCells_, ltsWave_);
    }
    enum class SmoothArm { undecided, host, device };
    PackedUpload up_;   // mesh.update()'s uploads, one transfer each (milestone K)
    SmoothArm smoothArm_ = SmoothArm::undecided;

    // ---- setRDeltaT.H (milestone I): after centralCourantNo.H, before ++runTime and the solves ----------------------
    // rDeltaT = max(1/maxDeltaT, surfaceSum(amaxSf)/((2*maxCo)*V)); rDeltaT.correctBoundaryConditions() (its patches are
    // extrapolatedCalculated or constraints: nothing reads them in the step); fvc::smooth(rDeltaT, coeff), the
    // FaceCellWave in its own order (device_fvc_smooth.cuh, to the bit of the host oracle); "Flow time scale min/max"
    // = gMin/gMax(1/rDeltaT). maxCo and maxDeltaT are controlDict's (readTimeControls.H), deltaT is not.
    const scalar* setRDeltaT(const FluxesIn& Fx)
    {
        const Case& cs = cs_;
        const int nC = cs.nC;
        if (ltsCells_.start.empty())
        {
            ltsCells_ = cellFaces(cs.m);
            rDT_.resize(nC);
            sa_.resize(nC);
            rDThost_.resize(static_cast<std::size_t>(nC));
        }
        surfaceSumKernel<<<blocksFor(nC), kTPB>>>(nC, dm_.ownerStart.data(), dm_.losort.data(), dm_.losortStart.data(),
                                                  dm_.bndCellStart.data(), dm_.bndPerm.data(), dm_.bndIsEmpty.data(),
                                                  Fx, sa_.data());
        const TimeControls& tc = cs.timeControls;
        rDeltaTKernel<<<blocksFor(nC), kTPB>>>(nC, 1/tc.maxDeltaT, 2*tc.maxCo, sa_.data(), dm_.V.data(), rDT_.data());
        cudaCheck(cudaGetLastError(), "rhoCentral rDeltaT");
        rDT_.copyTo(rDThost_);
        if (control_ != controlLtsUnsmoothed) smoothRDeltaT();
        scalar mn = 0, mx = 0;
        for (std::size_t c = 0; c < rDThost_.size(); ++c)
        {
            const scalar t = 1/rDThost_[c];
            if (c == 0 || t < mn) mn = t;
            if (c == 0 || t > mx) mx = t;
        }
        flowMin_[pending_] = mn;
        flowMax_[pending_] = mx;
        rDT_.copyFrom(rDThost_);
        return rDT_.data();
    }
    // the RAS model's fields (AUTO_WRITE: k, omega, nut, alphat), cells and patch values
    bool turbulenceFields(TurbulenceOut& o) const override
    {
        if (!turb_) return false;
        turb_->k().copyTo(o.k);
        turb_->omega().copyTo(o.omega);
        turb_->nut().copyTo(o.nut);
        turb_->nutB().copyTo(o.nutB);
        turb_->alphat().copyTo(o.alphat);
        turb_->alphatB().copyTo(o.alphatB);
        DeviceBuffer<scalar> b;
        turb_->kB(b);
        b.copyTo(o.kB);
        turb_->omegaB(b);
        b.copyTo(o.omegaB);
        return true;
    }
    // the rDeltaT the last step used, as setRDeltaT.H left it (written at write times: AUTO_WRITE)
    const std::vector<scalar>* rDeltaT() const override { return cs_.lts ? &rDThost_ : nullptr; }
    bool maxwellSlipState(std::vector<vector>& refValue, std::vector<scalar>& valueFraction) const override
    {
        if (!slip_ || !slip_->maxwell()) return false;
        slip_->maxwellState(refValue, valueFraction);
        return true;
    }

    // ---- the nozzle boundaries (milestone H), where OpenFOAM's updateCoeffs runs -----------------------------------
    // waveTransmissive reads p.oldTime().boundaryField(): the patch values as the step starts
    void keepOldPressureBoundary()
    {
        if (cs_.wtFaces.empty()) return;
        cudaCheck(cudaMemcpyAsync(pBold_.data(), F_.pB.data(), cs_.nB*sizeof(scalar), cudaMemcpyDeviceToDevice,
                                  cudaStreamPerThread), "p_b old");
    }
    // the flux the total conditions switch on (the control nozzleFlowInverted: -phi), the velocity whose dynamic head
    // they take (the control noDynamicHead: none)
    const DeviceBuffer<scalar>& nozzlePhi()
    {
        if (control_ != controlNozzleFlowInverted) return X_.phiB;
        negateKernel<<<blocksFor(cs_.nB), kTPB>>>(cs_.nB, X_.phiB.data(), phiInv_.data());
        cudaCheck(cudaGetLastError(), "rhoCentral -phi (nozzle)");
        return phiInv_;
    }
    // totalTemperature: Tw.evaluate() inside e.correctBoundaryConditions() -- this step's phi, U_b, the psi_b as it is
    void updateTotalTemperature()
    {
        if (cs_.ttFaces.empty()) return;
        const bool head = (control_ != controlNoDynamicHead);
        deviceUpdateTotalTemperature(dbT_, ttFaces_, ttT0_, ttGamma_, nozzlePhi(), head ? F_.UxB : zeroB_,
                                     head ? F_.UyB : zeroB_, head ? F_.UzB : zeroB_, F_.psiB);
    }
    // :287 p.correctBoundaryConditions(): totalPressure on this step's phi, U_b and the corrected psi_b;
    // waveTransmissive on p_b.oldTime, phi/rho_b (rho_b not yet updated, :288) and psi_b
    void updatePressureBoundaries(scalar deltaT)
    {
        if (!cs_.tpFaces.empty())
        {
            const bool head = (control_ != controlNoDynamicHead);
            deviceUpdateTotalPressurePsi(dbP_, tpFaces_, tpP0_, tpGamma_, nozzlePhi(), head ? F_.UxB : zeroB_,
                                         head ? F_.UyB : zeroB_, head ? F_.UzB : zeroB_, F_.psiB);
            if (tpRhoAny_)   // the rho form's faces (milestone K), on rho's patch values as they stand
            {
                const int n = static_cast<int>(cs_.tpFaces.size());
                totalPressureRhoKernel<<<blocksFor(n), kTPB>>>(n, tpFaces_.data(), tpRho_.data(), tpP0_.data(),
                                                               nozzlePhi().data(), (head ? F_.UxB : zeroB_).data(),
                                                               (head ? F_.UyB : zeroB_).data(),
                                                               (head ? F_.UzB : zeroB_).data(), F_.rhoB.data(),
                                                               dbP_.refValue.data());
                cudaCheck(cudaGetLastError(), "totalPressure, rho form");
            }
        }
        if (!cs_.wtFaces.empty())
            deviceUpdateWaveTransmissive(dbP_, wtFaces_, wtFieldInf_, wtLInf_, wtGamma_, pBold_, X_.phiB, F_.rhoB,
                                         F_.psiB, deltaT, control_ != controlWaveVolumeFlux);
    }

    // :195's phi decides every inletOutlet face of U, T and p before its evaluation (inletOutletFvPatchField::
    // updateCoeffs); the controls feed the switch another flux
    void switchInletOutlet()
    {
        const Case& cs = cs_;
        const int nB = cs.nB;
        // pressureInletOutletVelocity (milestone K): U.correctBoundaryConditions() at :232 updates it on this step's phi
        // (directionMixed, valueFraction neg(phi)*(I - nn)), on the cells' U as the explicit update left it
        if (cs.piovU && control_ != controlMotionPiovStatic)
            deviceUpdatePressureInletOutletVelocity(dbU_, X_.phiB, F_.Ux, F_.Uy, F_.Uz, true);
        if (!(cs.ioU || cs.ioP || cs.ioT)) return;
        const DeviceBuffer<scalar>* phiSwitch = &X_.phiB;
        const cudaStream_t q = cudaStreamPerThread;
        if (control_ == controlInletOutletInverted)
        {
            negateKernel<<<blocksFor(nB), kTPB>>>(nB, X_.phiB.data(), phiCtrl_.data());
            cudaCheck(cudaGetLastError(), "rhoCentral -phi");
            phiSwitch = &phiCtrl_;
        }
        else if (control_ == controlLaggedPhi)
        {
            if (!lagPrimed_)   // the first step has no step before: its own
                cudaCheck(cudaMemcpyAsync(phiCtrl_.data(), X_.phiB.data(), nB*sizeof(scalar),
                                          cudaMemcpyDeviceToDevice, q), "lagged phi");
            lagPrimed_ = true;
            phiSwitch = &phiCtrl_;
        }
        if (cs.ioU) deviceUpdateInletOutlet(dbU_, *phiSwitch);
        if (cs.ioT) deviceUpdateInletOutlet(dbT_, *phiSwitch);
        if (cs.ioP) deviceUpdateInletOutlet(dbP_, *phiSwitch);
        if (control_ == controlLaggedPhi)   // what the next step will switch on
            cudaCheck(cudaMemcpyAsync(phiCtrl_.data(), X_.phiB.data(), nB*sizeof(scalar),
                                      cudaMemcpyDeviceToDevice, q), "lagged phi");
    }

    // U.correctBoundaryConditions(): the shared boundary's faces, then the transform faces from the cells
    // update: maxwellSlipU's updateCoeffs runs first (U.correctBoundaryConditions() at :232); after the U solve the
    // patch is updated() and only evaluates (milestone I)
    void evaluateU(bool update = false)
    {
        const int nB = cs_.nB;
        if (!nB) return;
        DeviceFields& F = F_;
        if (slip_ && slip_->maxwell())
        {
            if (update) slip_->updateMaxwell(F.T, F.TB, F.rhoB, F.psiB, vis_->tauB());
            slip_->emulate(dbU_, F.Ux, F.Uy, F.Uz);
        }
        deviceBCValue(dbU_.comp[0], F.Ux, F.UxB);
        deviceBCValue(dbU_.comp[1], F.Uy, F.UyB);
        deviceBCValue(dbU_.comp[2], F.Uz, F.UzB);
        uTransformKernel<<<blocksFor(nB), kTPB>>>(nB, control_, bCell_.data(), uTransform_.data(), uR_.data(),
                                                  F.Ux.data(), F.Uy.data(), F.Uz.data(), F.UxB.data(), F.UyB.data(),
                                                  F.UzB.data());
        cudaCheck(cudaGetLastError(), "rhoCentral U transform faces");
        if (slip_) slip_->evaluate(F.Ux, F.Uy, F.Uz, F.UxB, F.UyB, F.UzB);
    }

    // e.correctBoundaryConditions() (Tw.evaluate() on the cells' T given), thermo.correct()'s patch half, rhoU_b and
    // rhoE_b: what the inviscid step does after its cell kernel
    // totalT: Tw.evaluate() runs (e.correctBoundaryConditions() with e's patches not yet updated). After the e solve
    // it does not: the matrix's construction left them updated(), so the solve's correctBoundaryConditions() only
    // evaluates (fvPatchField::evaluate), and T_b stays the construction's.
    void patchUpdate(const DeviceBuffer<scalar>& Tcells, bool totalT = true, bool countInletOutlet = true)
    {
        const int nB = cs_.nB;
        if (!nB) return;
        DeviceFields& F = F_;
        if (totalT) updateTotalTemperature();
        // smoluchowskiJumpT: mixedEnergy's updateCoeffs -> Tw.evaluate() at :265 on mu_b, rho_b, psi_b as they stand;
        // after the e solve T_b stays what the matrix's construction evaluated (milestone I)
        if (slip_ && totalT) slip_->updateSmoluchowski(F.TB, F.rhoB, F.psiB, F.pB, dbT_);
        deviceBCValue(dbT_, Tcells, TbEval_);
        if (slip_ && !totalT) slip_->restoreTb(TbEval_);
        const PatchState B{F.rhoB.data(), F.rhoUxB.data(), F.rhoUyB.data(), F.rhoUzB.data(), F.rhoEB.data(),
                           F.eB.data(), F.TB.data(), F.psiB.data(), F.UxB.data(), F.UyB.data(), F.UzB.data(),
                           F.pB.data(), TbEval_.data()};
        const PatchTransforms X{F.rho.data(), F.Ux.data(), F.Uy.data(), F.Uz.data(), uTransform_.data(),
                                uR_.data(), rhoConstraint_.data(), TseedOwn_.data()};
        patchKernel<<<blocksFor(nB), kTPB>>>(nB, pending_, cs_.gas, control_, bCell_.data(), TfixesValue_.data(),
                                             F.e.data(), B, X, status_.data(), X_.phiB.data(), ioFace_.data(),
                                             countInletOutlet ? ioCount_.data() : nullptr);
        cudaCheck(cudaGetLastError(), "rhoCentral patches");
    }

    // the viscous step (milestone G): rhoCentralFoam.C:102-288 with the correctors, see rhoCentralViscous.cuh
    // turbulence->correct() (rhoCentralFoam.C:290): on this step's rho, U, the central mass flux phi (:195) and the
    // thermo's mu of the last thermo.correct(); fvm::ddt(rho, k|omega) on rho.oldTime(), the rho the step started
    // from, with the case's ddt scheme; its omega and k solve lines in the step's report
    void correctTurbulence(scalar rDeltaT, StepReport& rep)
    {
        DeviceFields& F = F_;
        vis_->laminarMu(F, muLamC_, muLamB_);
        gpu::compressibleRAS::Flow fl;
        fl.rho = &F.rho;
        fl.rhoB = &F.rhoB;
        fl.rhoOld = &vis_->rhoOld();
        fl.Ux = &F.Ux;
        fl.Uy = &F.Uy;
        fl.Uz = &F.Uz;
        fl.mu = &muLamC_;
        fl.muB = &muLamB_;
        fl.phiInt = &X_.phi;
        fl.phiBnd = &X_.phiB;
        fl.rDeltaT = rDeltaT;
        fl.rDeltaTCells = cs_.lts ? &rDT_ : nullptr;
        fl.controlPhiUndivided = (control_ == controlRasPhiUndivided);
        turb_->correct(fl, dbU_);
        const auto& r = turb_->residuals();
        const DeviceSolverPerf* perf[2] = {&r.omegaPerf, &r.kPerf};
        const char* nm[2] = {"omega", "k"};   // kOmegaSSTBase::correct solves omega, then k
        for (int j = 0; j < 2 && rep.nSolves < 6; ++j)
        {
            SolveLine& l = rep.solves[rep.nSolves++];
            std::strncpy(l.field, nm[j], sizeof(l.field) - 1);
            l.initialResidual = perf[j]->initialResidual;
            l.finalResidual = perf[j]->finalResidual;
            l.nIterations = perf[j]->nIterations;
        }
    }

    void stepViscous(scalar deltaT)
    {
        const Case& cs = cs_;
        DeviceFields& F = F_;
        const int nC = cs.nC, nB = cs.nB;
        const scalar rDeltaT = 1.0/deltaT;
        StepReport& rep = solveRep_[pending_];
        rep = StepReport{};
        keepOldPressureBoundary();

        std::optional<interPhase::Nested> ph;
        ph.emplace("step: fluxes, beginStep");
        centralFluxes(dm_, cs, control_, F, X_, outflow_.data(), deltaT, meshPhiI(), meshPhiB());   // :102-210
        vis_->beginStep(F, dbU_);                                            // :220-221, the old fields
        const FluxesIn Fx{X_.phi.data(), X_.phiUx.data(), X_.phiUy.data(), X_.phiUz.data(), X_.phiEp.data(),
                          X_.amaxSf.data(), X_.phiB.data(), X_.phiUxB.data(), X_.phiUyB.data(), X_.phiUzB.data(),
                          X_.phiEpB.data(), X_.amaxSfB.data()};
        const CellState S{F.rho.data(), F.rhoUx.data(), F.rhoUy.data(), F.rhoUz.data(), F.rhoE.data(), F.e.data(),
                          F.T.data(), Told_.data(), F.p.data(), F.psi.data(), F.Ux.data(), F.Uy.data(), F.Uz.data()};
        ph.reset();
        ph.emplace("step: setRDeltaT.H, momentum, U's BCs");
        const scalar* rDT = cs.lts ? setRDeltaT(Fx) : nullptr;   // :186-190 (milestone I)
        momentumKernel<<<blocksFor(nC), kTPB>>>(nC, rDeltaT, cs.sd.valid(0), cs.sd.valid(1), cs.sd.valid(2),
                                                dm_.ownerStart.data(), dm_.losort.data(), dm_.losortStart.data(),
                                                dm_.bndCellStart.data(), dm_.bndPerm.data(), dm_.bndIsEmpty.data(),
                                                dm_.V.data(), Fx, S, coCell_.data(), divPhiEp_.data(), rDT, V0());
        cudaCheck(cudaGetLastError(), "rhoCentral momentum");
        if (nB) switchInletOutlet();
        evaluateU(true);                                                      // :232
        if (turb_ && control_ == controlRasCorrectFirst) correctTurbulence(rDeltaT, rep);   // the control
        ph.reset();
        ph.emplace("step: solve U (assembly, smoothSolver)");
        vis_->solveU(F, dbU_, rDeltaT, rep, cs.lts ? &rDT_ : nullptr);        // :234-244
        ph.reset();
        ph.emplace("step: energy explicit, T's BCs");
        evaluateU();                                                          // fvMatrix::solve's U.correctBC
        const DeviceBuffer<scalar>& dS = vis_->divSigmaDotU(F, dbU_, X_);     // :246-253
        energyKernel<<<blocksFor(nC), kTPB>>>(nC, pending_, rDeltaT, cs.gas, dm_.V.data(), divPhiEp_.data(),
                                              control_ == controlNoSigmaDotU ? nullptr : dS.data(), S,
                                              status_.data(), rDT, V0());     // :258-266
        cudaCheck(cudaGetLastError(), "rhoCentral energy");
        patchUpdate(control_ == controlInletOutletNewT ? F.T : Told_);        // :265-271
        // fvMatrix's constructor updates e's patches (fixedEnergy::updateCoeffs -> Tw.evaluate()) on the psi_b the first
        // thermo.correct() left: a totalTemperature face's T_b moves, and its he_b with it
        if (!cs.ttFaces.empty() && control_ != controlTotalTemperatureOnce)
        {
            updateTotalTemperature();
            const int n = static_cast<int>(cs.ttFaces.size());
            fixedEnergyFaceKernel<<<blocksFor(n), kTPB>>>(n, ttFaces_.data(), cs.gas, dbT_.refValue.data(), F.pB.data(),
                                                          F.eB.data());
            cudaCheck(cudaGetLastError(), "fixedEnergy (totalTemperature)");
        }
        // smoluchowskiJumpT at the e matrix's construction (mixedEnergy::updateCoeffs): the valueFraction on the psi_b
        // and mu(T_b) the first thermo.correct() left, T_b evaluated on the cells' new T and kept (milestone I)
        if (slip_ && slip_->smoluchowski())
        {
            slip_->updateSmoluchowski(F.TB, F.rhoB, F.psiB, F.pB, dbT_);
            deviceBCValue(dbT_, F.T, TbEval_);
            slip_->keepTb(TbEval_);
        }
        // e's matrix: its construction's updateCoeffs ran Tw.evaluate() on every T patch (fixed-, gradient- and
        // mixedEnergy alike), on the cells' T of the first thermo.correct(), BEFORE turbulence->alphaEff() was called
        // -- the order OpenFOAM's build evaluates `fvm::ddt(rho, e) - fvc::ddt(rho, e) - fvm::laplacian(alphaEff, e)`
        // in, MEASURED (biconic25-55Run35, one step: e's initial residual 4.4e-7 off with CpByCpv on the old T_b,
        // 3e-14 with it on these). alphahe() = CpByCpv()*alpha_: CpByCpv's patch values read these T_b, alpha_'s
        // the thermo.correct()'s. Only a T-dependent CpByCpv (janaf) can tell the two apart (milestone I).
        if (cs.gas.janaf && nB && control_ != controlAlphaheOldT)
        {
            deviceBCValue(dbT_, F.T, TbGamma_);
            if (slip_) slip_->restoreTb(TbGamma_);
            vis_->setAlphaheGammaT(&TbGamma_);
        }
        ph.reset();
        ph.emplace("step: solve e (assembly, smoothSolver)");
        vis_->solveE(F, dbT_, rDeltaT, rep, cs.lts ? &rDT_ : nullptr);        // :273-277
        ph.reset();
        ph.emplace("step: thermo, p's BCs, turbulence, Courant");
        patchUpdate(F.T, false, false);                                       // the solve's e.correctBC, :278-280
        thermoPressureKernel<<<blocksFor(nC), kTPB>>>(nC, pending_, cs.gas, S, status_.data());   // :278-286
        cudaCheck(cudaGetLastError(), "rhoCentral thermo, p");
        if (nB)
        {
            updatePressureBoundaries(deltaT);
            deviceBCValue(dbP_, F.p, F.pB);                                   // :287
            rhoBKernel<<<blocksFor(nB), kTPB>>>(nB, F.psiB.data(), F.pB.data(), F.rhoB.data());   // :288
            cudaCheck(cudaGetLastError(), "rhoCentral rho_b");
        }
        // turbulence->correct() (:290): after p, p_b and rho_b, on this step's rho, U, the central mass flux phi
        // (:195) and the thermo's mu of the last thermo.correct()
        if (turb_ && control_ != controlRasCorrectFirst) correctTurbulence(rDeltaT, rep);
        courantKernel<<<1, 1024>>>(nC, coCell_.data(), dm_.V.data(), courant_.data() + 3*pending_);
        cudaCheck(cudaGetLastError(), "rhoCentral courant");
        vis_->boundaryInflow(F, deltaT, visIn_);
        ph.reset();
        deltaT_[pending_] = deltaT;
        ++pending_;
    }

    std::vector<StepReport> flush() override
    {
        std::vector<StepReport> out(pending_);
        if (!pending_) return out;
        const double* co = hostCourant_;
        const int* st = hostStatus_;
        const cudaStream_t q = cudaStreamPerThread;   // the stream the kernels run on (--default-stream per-thread)
        cudaCheck(cudaMemcpyAsync(hostCourant_, courant_.data(), 3*pending_*sizeof(double), cudaMemcpyDeviceToHost, q),
                  "courant D2H");
        cudaCheck(cudaMemcpyAsync(hostStatus_, status_.data(), pending_*sizeof(int), cudaMemcpyDeviceToHost, q),
                  "status D2H");
        cudaCheck(cudaMemsetAsync(status_.data(), 0, pending_*sizeof(int), q), "status reset");
        cudaCheck(cudaStreamSynchronize(q), "reports");
        for (int i = 0; i < pending_; ++i)
        {
            if (cs_.nIF)
            {
                out[i].CoNum = 0.5*co[3*i]*deltaT_[i];
                out[i].meanCoNum = 0.5*(co[3*i + 1]/co[3*i + 2])*deltaT_[i];
            }
            out[i].status = st[i];
            out[i].flowMin = flowMin_[i];
            out[i].flowMax = flowMax_[i];
            out[i].nSolves = solveRep_[i].nSolves;
            for (int j = 0; j < solveRep_[i].nSolves; ++j) out[i].solves[j] = solveRep_[i].solves[j];
        }
        pending_ = 0;
        return out;
    }
    int pending() const override { return pending_; }
    int window() const override { return kReportWindow; }

    void download(State& s) override
    {
        const DeviceFields& F = F_;
        s.rho = F.rho.host();
        downCmpt(F.rhoUx, s.rhoU, 0);
        downCmpt(F.rhoUy, s.rhoU, 1);
        downCmpt(F.rhoUz, s.rhoU, 2);
        s.rhoE = F.rhoE.host();
        s.e = F.e.host();
        s.T = F.T.host();
        s.p = F.p.host();
        s.psi = F.psi.host();
        downCmpt(F.Ux, s.U, 0);
        downCmpt(F.Uy, s.U, 1);
        downCmpt(F.Uz, s.U, 2);
        s.rhoB = F.rhoB.host();
        downCmpt(F.rhoUxB, s.rhoUB, 0);
        downCmpt(F.rhoUyB, s.rhoUB, 1);
        downCmpt(F.rhoUzB, s.rhoUB, 2);
        s.rhoEB = F.rhoEB.host();
        s.eB = F.eB.host();
        s.TB = F.TB.host();
        s.pB = F.pB.host();
        s.psiB = F.psiB.host();
        downCmpt(F.UxB, s.UB, 0);
        downCmpt(F.UyB, s.UB, 1);
        downCmpt(F.UzB, s.UB, 2);
    }

    void boundaryOutflow(scalar& mass, vector& mom, scalar& energy) const override
    {
        const std::vector<double> r = outflow_.host();
        mass = r[0];
        mom = vector{r[1], r[2], r[3]};
        energy = r[4];
        if (vis_)   // the viscous fluxes through the patches, into the domain: out of it is their negative
        {
            const std::vector<double> v = visIn_.host();
            mom = vector{mom.x - v[0], mom.y - v[1], mom.z - v[2]};
            energy -= v[3];
        }
    }

    void inletOutletInflow(long long& inflow, long long& faceSteps) const override
    {
        const std::vector<unsigned long long> r = ioCount_.host();
        inflow = static_cast<long long>(r[0]);
        faceSteps = static_cast<long long>(r[1]);
    }

    std::size_t deviceBytes() const override { return bytes_; }

    void solveResidualL1(vector& mom, scalar& energy) const override
    {
        double r[4] = {0, 0, 0, 0};
        if (vis_) vis_->solveResidualL1(r);
        mom = vector{r[0], r[1], r[2]};
        energy = r[3];
    }
    void solveIdentity(vector& mom, scalar& energy) const override
    {
        double r[4] = {0, 0, 0, 0};
        if (vis_) vis_->solveIdentity(r);
        mom = vector{r[0], r[1], r[2]};
        energy = r[3];
    }

    void solveResidual(vector& mom, scalar& energy) const override
    {
        double r[4] = {0, 0, 0, 0};
        if (vis_) vis_->solveResidual(r);
        mom = vector{r[0], r[1], r[2]};
        energy = r[3];
    }

    ~DeviceArm() override
    {
        cudaFreeHost(hostCourant_);
        cudaFreeHost(hostStatus_);
    }
    // mesh.update() (rhoCentralFoam.C:96, after ++runTime): dynamicMotionSolverFvMesh::update -- the motion on the
    // host (brae's shared DynamicMotionSolverFvMesh: newPoints, V0, oldPoints, meshPhi, the geometry and the patches in
    // place) -- then what the arm holds of the geometry, then U.correctBoundaryConditions() as update() ends.
    // Returns the motion solver's solve for the log line.
    SolverPerformance moveMesh(scalar time, scalar deltaT, label timeIndex) override
    {
        const Case& cs = cs_;
        DynamicMotionSolverFvMesh& mo = *cs.motion;
        std::optional<interPhase::Nested> ph;
        ph.emplace("motion: DynamicMotionSolverFvMesh::update");
        mo.update(time, deltaT, timeIndex);
        moved_ = mo.moving();
        ph.reset();
        ph.emplace("motion: refreshDeviceMeshGeometry");
        // the device mesh's areas, volumes, centres, weights and deltaCoeffs; the boundaries' own areas and deltaCoeffs
        // (and the normals pressureInletOutletVelocity and a symmetry gradient read), from the same builders
        refreshDeviceMeshGeometryInPlace(dm_, cs.m, cs.g, cs.patches, &up_);   // the buffers kept: no cache keys on them
        ph.reset();
        ph.emplace("motion: boundaries' geometry rebuilt");
        // the arrays a fresh build of U's, p's and T's boundaries would give (deviceBoundaryGeometry: the builders'
        // layout and arithmetic), copied over theirs; the scalar boundaries carry no gradSymN
        {
            const DeviceBoundaryGeometryHost bg = deviceBoundaryGeometry(cs.patches, cs.g);
            for (int c = 0; c < 3; ++c)
            {
                up_.add(dbU_.comp[c].deltaCoeffs, bg.dc);
                up_.add(dbU_.comp[c].magSf, bg.ms);
                up_.add(dbU_.comp[c].gradSymN, bg.gsn);
            }
            up_.add(dbU_.nx, bg.nrm[0]);
            up_.add(dbU_.ny, bg.nrm[1]);
            up_.add(dbU_.nz, bg.nrm[2]);
            for (DeviceBoundary* d : {&dbP_, &dbT_})
            {
                up_.add(d->deltaCoeffs, bg.dc);
                up_.add(d->magSf, bg.ms);
            }
            up_.flush();
        }
        ph.reset();
        ph.emplace("motion: V0, meshPhi up");
        // V0 (fvMesh::V0, stored before the move) and meshPhi, flat over the boundary in patch order (zero on empty)
        V0d_.copyFrom(mo.V0());
        meshPhiId_.copyFrom(mo.meshPhi().internal);
        std::vector<scalar> mb(static_cast<std::size_t>(cs.nB), 0.0);
        for (label k = 0; k < cs.nB; ++k)
        {
            if (cs.bEmpty[k]) continue;
            const int pi = cs.bPatch[k];
            const label i = cs.nIF + k - cs.patches[static_cast<std::size_t>(pi)].start;
            mb[static_cast<std::size_t>(k)] = mo.meshPhi().boundary[static_cast<std::size_t>(pi)][static_cast<std::size_t>(i)];
        }
        meshPhiBd_.copyFrom(mb);
        ph.reset();
        ph.emplace("motion: viscous geometry");
        if (vis_)
        {
            vis_->setMovingV0(control_ == controlMotionDdtStatic ? nullptr : &V0d_);
            vis_->geometryChanged();
        }
        ph.reset();
        ph.emplace("motion: movingWallVelocity, U's BCs, gcl");
        movingWallVelocity(deltaT, mb);
        // U.correctBoundaryConditions() (dynamicMotionSolverFvMesh.C, update): every U patch's updateCoeffs on the phi
        // that exists now -- the last step's, or createFields' fvc::flux(rhoU) before the first -- then evaluate
        if (cs.piovU && control_ != controlMotionPiovStatic)
            deviceUpdatePressureInletOutletVelocity(dbU_, X_.phiB, F_.Ux, F_.Uy, F_.Uz, true);
        if (cs.ioU) deviceUpdateInletOutlet(dbU_, X_.phiB);
        evaluateU();
        gcl(deltaT, mo);
        return mo.velocityComponent() ? mo.velocityComponent()->lastSolve() : SolverPerformance{};
    }
    scalar gclMax() const override { return gclMax_; }

    DeviceArm(const DeviceArm&) = delete;
    DeviceArm& operator=(const DeviceArm&) = delete;

private:
    // movingWallVelocityFvPatchVectorField::Uwall(): Up = (face::centre(points) - face::centre(oldPoints))/deltaT,
    // Un = meshPhi_b/(magSf + VSMALL), Uwall = Up + n*(Un - (n & Up)), each field operation its own loop (no product
    // contracted into a sum); written into the fixedValue faces' refValue (the patch is a fixedValue whose value the
    // mesh motion sets: updateCoeffs assigns it when mesh.moving())
    void movingWallVelocity(scalar deltaT, const std::vector<scalar>& meshPhiFlat)
    {
        const Case& cs = cs_;
        const DynamicMotionSolverFvMesh& mo = *cs.motion;
        const std::size_t n = cs.mwFaces.size();
        if (!n) return;
        std::vector<vector> Up(n), nf(n), ns(n);
        std::vector<scalar> Un(n), d(n);
        for (std::size_t j = 0; j < n; ++j)
        {
            const label f = cs.nIF + cs.mwFaces[j];
            const vector nc = faceCentreOfPoints(cs.m, f, cs.m.points());
            const vector oc = faceCentreOfPoints(cs.m, f, mo.oldPoints());
            const vector dc{nc.x - oc.x, nc.y - oc.y, nc.z - oc.z};
            Up[j] = vector{dc.x/deltaT, dc.y/deltaT, dc.z/deltaT};
            const vector S = cs.g.Sf()[static_cast<std::size_t>(f)];
            const scalar a = cs.g.magSf()[static_cast<std::size_t>(f)];
            nf[j] = vector{S.x/a, S.y/a, S.z/a};
            const scalar den = a + 1.0e-300;   // magSf + VSMALL
            Un[j] = meshPhiFlat[static_cast<std::size_t>(cs.mwFaces[j])]/den;
        }
        for (std::size_t j = 0; j < n; ++j)   // Un - (n & Up)
        {
            const scalar nUp = nf[j].x*Up[j].x + nf[j].y*Up[j].y + nf[j].z*Up[j].z;
            d[j] = Un[j] - nUp;
        }
        for (std::size_t j = 0; j < n; ++j)   // n*(...)
        {
            volatile scalar x = nf[j].x*d[j], y = nf[j].y*d[j], z = nf[j].z*d[j];
            ns[j] = vector{x, y, z};
        }
        for (std::size_t j = 0; j < n; ++j)   // Up + ...
        {
            const vector w{Up[j].x + ns[j].x, Up[j].y + ns[j].y, Up[j].z + ns[j].z};
            const std::size_t k = static_cast<std::size_t>(cs.mwFaces[j]);
            if (control_ == controlMotionWallStatic)   // the control: the wall's velocity left at zero
            {
                refUhost_[0][k] = refUhost_[1][k] = refUhost_[2][k] = 0;
                continue;
            }
            refUhost_[0][k] = w.x;
            refUhost_[1][k] = w.y;
            refUhost_[2][k] = w.z;
        }
        for (int c = 0; c < 3; ++c) dbU_.comp[c].refValue.copyFrom(refUhost_[c]);
    }

    // the discrete space-conservation law the swept volumes make: V - V0 = deltaT*sum(meshPhi) per cell (a diagnostic)
    void gcl(scalar deltaT, const DynamicMotionSolverFvMesh& mo)
    {
        const Case& cs = cs_;
        const std::vector<scalar>& V = cs.g.V();
        const std::vector<scalar>& V0 = mo.V0();
        std::vector<scalar> s(static_cast<std::size_t>(cs.nC), 0.0);
        const std::vector<label>& own = cs.m.owner();
        const std::vector<label>& nei = cs.m.neighbour();
        for (label f = 0; f < cs.nIF; ++f)
        {
            s[static_cast<std::size_t>(own[f])] += mo.meshPhi().internal[static_cast<std::size_t>(f)];
            s[static_cast<std::size_t>(nei[f])] -= mo.meshPhi().internal[static_cast<std::size_t>(f)];
        }
        for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
        {
            const FvPatch& p = cs.patches[pi];
            if (p.type == "empty") continue;
            for (label i = 0; i < p.size; ++i)
                s[static_cast<std::size_t>(p.faceCells[static_cast<std::size_t>(i)])] +=
                    mo.meshPhi().boundary[pi][static_cast<std::size_t>(i)];
        }
        for (label c = 0; c < cs.nC; ++c)
        {
            const std::size_t i = static_cast<std::size_t>(c);
            const scalar r = std::fabs(V[i] - V0[i] - deltaT*s[i])/V[i];
            if (r > gclMax_) gclMax_ = r;
        }
    }

    const Case& cs_;
    int control_;
    DeviceMesh dm_;
    DeviceFields F_;
    DeviceFluxes X_;
    DeviceVectorBoundary dbU_;
    DeviceBoundary dbP_, dbT_;
    DeviceBuffer<int> TfixesValue_;
    DeviceBuffer<label> bCell_;
    DeviceBuffer<int> uTransform_;
    DeviceBuffer<tensor> uR_;
    DeviceBuffer<char> rhoConstraint_, TseedOwn_, ioFace_;
    DeviceBuffer<unsigned long long> ioCount_;
    DeviceBuffer<scalar> phiCtrl_;   // the controls' switch flux: -phi, or the step before's phi
    bool lagPrimed_ = false;
    DeviceBuffer<scalar> Told_, TbEval_, coCell_;
    std::unique_ptr<ViscousCorrectors> vis_;   // mu > 0 (milestone G)
    std::unique_ptr<SlipWalls> slip_;          // maxwellSlipU and smoluchowskiJumpT (milestone I)
    std::unique_ptr<gpu::compressibleRAS::KOmegaSST> turb_;   // RAS (milestone J)
    // the moving mesh (milestone K): the host fields the boundaries are rebuilt from, V0 and meshPhi as the kernels
    // read them (null until the mesh has moved: polyMesh::moving()), the movingWallVelocity faces' host values, and the
    // discrete space-conservation residual of every step
    PatchFields* pf_ = nullptr;
    DeviceBuffer<scalar> V0d_, meshPhiId_, meshPhiBd_;
    std::vector<scalar> refUhost_[3];
    bool moved_ = false;
    scalar gclMax_ = 0;
    const scalar* V0() const { return moved_ && control_ != controlMotionDdtStatic ? V0d_.data() : nullptr; }
    const scalar* meshPhiI() const
    {
        return moved_ && cs_.nIF && control_ != controlMotionFluxAbsolute ? meshPhiId_.data() : nullptr;
    }
    const scalar* meshPhiB() const
    {
        return moved_ && cs_.nB && control_ != controlMotionFluxAbsolute ? meshPhiBd_.data() : nullptr;
    }
    DeviceBuffer<scalar> muLamC_, muLamB_;
    DeviceBuffer<scalar> TbGamma_;
    DeviceBuffer<label> tpFaces_, ttFaces_, wtFaces_;   // the nozzle boundaries (milestone H)
    DeviceBuffer<char> tpRho_;                          // totalPressure's rho-form faces (milestone K)
    bool tpRhoAny_ = false;
    DeviceBuffer<scalar> tpP0_, tpGamma_, ttT0_, ttGamma_, wtFieldInf_, wtLInf_, wtGamma_, pBold_, zeroB_, phiInv_;
    DeviceBuffer<scalar> divPhiEp_;
    DeviceBuffer<double> visIn_;
    StepReport solveRep_[kReportWindow];
    // LTS (milestone I): the per-cell rDeltaT, its host copy for fvc::smooth's wave and the flow time scales
    DeviceBuffer<scalar> rDT_, sa_;
    std::vector<scalar> rDThost_;
    CellFaces ltsCells_;
    DeviceSmoothWave ltsWave_;
    scalar flowMin_[kReportWindow] = {};
    scalar flowMax_[kReportWindow] = {};
    DeviceBuffer<int> status_;
    DeviceBuffer<double> courant_, outflow_;
    scalar deltaT_[kReportWindow] = {};
    double* hostCourant_ = nullptr;
    int* hostStatus_ = nullptr;   // each pending step's deltaT, which its Courant numbers are scaled by
    int pending_ = 0;
    std::size_t bytes_ = 0;
};

} // namespace

std::unique_ptr<Arm> makeDeviceArm(const Case& cs, const State& s0, PatchFields& pf, int control)
{
    return std::make_unique<DeviceArm>(cs, s0, pf, control);
}

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
