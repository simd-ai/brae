// cf rotorDiskSource (Froude BEM), geometry build (host) + per-iteration body-force kernel. See rotor_disk.cuh.
// Mirrors OF rotorDiskSource::setFaceArea / calculate (single lookup profile, fixedTrim, no coning).
#include "rotor_disk.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace brae {
namespace {
constexpr int TPB = 128;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// Linear interpolation of (twist,chord) at a radius in the sorted blade table (OF bladeModel::interpolateWeights).
inline void bladeInterp(
    scalar r,
    const std::vector<scalar>& R,
    const std::vector<scalar>& tw,
    const std::vector<scalar>& ch,
    scalar& twist,
    scalar& chord)
{
    const int m = (int)R.size();
    if (m == 1)
    {
        twist = tw[0];
        chord = ch[0];
        return;
    }
    int i2 = 0;
    while (i2 < m && R[i2] < r) ++i2;
    if (i2 == 0)
    {
        twist = tw[0];
        chord = ch[0];
    }
    else if (i2 == m)
    {
        twist = tw[m-1];
        chord = ch[m-1];
    }
    else
    {
        const int i1 = i2-1;
        const scalar w = (r-R[i1])/(R[i2]-R[i1]);
        twist = tw[i1] + w*(tw[i2]-tw[i1]);
        chord = ch[i1] + w*(ch[i2]-ch[i1]);
    }
}

__device__ __forceinline__
scalar lookupCdl(
    scalar a,
    const scalar* A,
    const scalar* C,
    int n)
{
    // linear interpolation of C vs A at a (A sorted ascending); clamp outside the table (OF interpolateWeights)
    if (n == 1) return C[0];
    int i2 = 0;
    while (i2 < n && A[i2] < a) ++i2;
    if (i2 == 0) return C[0];
    if (i2 == n) return C[n-1];
    const int i1 = i2-1;
    const scalar w = (a-A[i1])/(A[i2]-A[i1]);
    return C[i1] + w*(C[i2]-C[i1]);
}

__device__
void rotorForceKernel(
    int n,
    const label* __restrict__ cells,
    vector axis,
    scalar omega,
    scalar nBlades,
    scalar tipEffect,
    scalar rMax,
    scalar theta0,
    int localInflow,
    vector inletVel,
    const scalar* __restrict__ e2x,
    const scalar* __restrict__ e2y,
    const scalar* __restrict__ e2z,
    const scalar* __restrict__ radius,
    const scalar* __restrict__ area,
    const scalar* __restrict__ twist,
    const scalar* __restrict__ chord,
    const scalar* __restrict__ pA,
    const scalar* __restrict__ pCd,
    const scalar* __restrict__ pCl,
    int nProf,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ fx,
    scalar* __restrict__ fy,
    scalar* __restrict__ fz)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar r = radius[i];
    if (!(r > 1e-12) || !(area[i] > 1e-30)) return;

    vector U = localInflow ? vector{Ux[c], Uy[c], Uz[c]} : inletVel;
    const vector e2{e2x[i], e2y[i], e2z[i]};
    scalar Uy_az = e2.x*U.x + e2.y*U.y + e2.z*U.z;           // e2 . U  (azimuthal)
    const scalar Uz_ax = axis.x*U.x + axis.y*U.y + axis.z*U.z;     // e3 . U  (axial)
    Uy_az = r*omega - Uy_az;                                 // blade-relative tangential velocity
    const scalar magSqr = Uy_az*Uy_az + Uz_ax*Uz_ax;
    scalar alphaGeom = theta0 + twist[i];
    if (omega < 0) alphaGeom = 3.14159265358979323846 - alphaGeom;
    scalar alphaEff = alphaGeom - atan2(-Uz_ax, Uy_az);
    if (alphaEff > 3.14159265358979323846)  alphaEff -= 6.28318530717958647692;
    if (alphaEff < -3.14159265358979323846) alphaEff += 6.28318530717958647692;
    const scalar Cd = lookupCdl(alphaEff, pA, pCd, nProf);
    const scalar Cl = lookupCdl(alphaEff, pA, pCl, nProf);
    const scalar tip = (r/rMax - tipEffect < 0) ? 1.0 : 0.0;
    const scalar pDyn = 0.5*magSqr;                          // rho = 1 (incompressible kinematic)
    const scalar f = pDyn*chord[i]*nBlades*area[i]/r/6.28318530717958647692;
    const scalar localAz = -f*Cd, localAx = tip*f*Cl;        // localForce = (0, -f*Cd, tip*f*Cl) in (e1,e2,e3)
    fx[c] = localAz*e2.x + localAx*axis.x;                   // transform(Rcyl, localForce)
    fy[c] = localAz*e2.y + localAx*axis.y;
    fz[c] = localAz*e2.z + localAx*axis.z;
}
} // namespace


DeviceRotorDisk buildDeviceRotorDisk(
    const RotorDiskParams& p,
    const std::vector<vector>& cellC,
    const std::vector<vector>& Sf,
    const std::vector<label>& owner,
    const std::vector<label>& neighbour,
    label nInternalFaces)
{
    DeviceRotorDisk d;
    if (!p.active || p.cells.empty()) return d;
    d.active = true;
    d.n = (int)p.cells.size();
    const scalar am = std::sqrt(p.axis.x*p.axis.x + p.axis.y*p.axis.y + p.axis.z*p.axis.z);
    d.axis = vector{p.axis.x/am, p.axis.y/am, p.axis.z/am};
    d.omega = p.omega;
    d.nBlades = (scalar)p.nBlades;
    d.tipEffect = p.tipEffect;
    d.theta0 = p.theta0;
    d.localInflow = p.localInflow;
    d.inletVel = p.inletVel;
    const std::size_t nC = cellC.size();
    std::vector<label> cellAddr(nC, -1);
    for (std::size_t i = 0; i < p.cells.size(); ++i)
        cellAddr[p.cells[i]] = (label)i;
    std::vector<scalar> e2x(d.n), e2y(d.n), e2z(d.n), rad(d.n), tw(d.n), ch(d.n);
    scalar rMax = 0;
    for (int i = 0; i < d.n; ++i)
    {
        const vector cc = cellC[p.cells[i]];
        const vector rel{cc.x-p.origin.x, cc.y-p.origin.y, cc.z-p.origin.z};
        const scalar ax = rel.x*d.axis.x + rel.y*d.axis.y + rel.z*d.axis.z;
        const vector radv{rel.x-ax*d.axis.x, rel.y-ax*d.axis.y, rel.z-ax*d.axis.z};   // radial component
        const scalar r = std::sqrt(radv.x*radv.x + radv.y*radv.y + radv.z*radv.z);
        rad[i] = r;
        rMax = std::max(rMax, r);
        const vector e1 = r > 1e-12 ? vector{radv.x/r, radv.y/r, radv.z/r} : vector{1,0,0};
        e2x[i] = d.axis.y*e1.z - d.axis.z*e1.y;                 // e2 = axis x e1 (azimuthal)
        e2y[i] = d.axis.z*e1.x - d.axis.x*e1.z;
        e2z[i] = d.axis.x*e1.y - d.axis.y*e1.x;
        scalar twist, chord;
        bladeInterp(r, p.bladeR, p.bladeTwist, p.bladeChord, twist, chord);
        tw[i] = twist;
        ch[i] = chord;
    }
    d.rMax = rMax;
    // setFaceArea: area[diskCell] = sum magSf of internal faces on the disk boundary with face-normal aligned (+axis).
    std::vector<scalar> area(d.n, 0.0);
    for (label f = 0; f < nInternalFaces; ++f)
    {
        const label o = owner[f], nb = neighbour[f];
        const label oa = cellAddr[o], na = cellAddr[nb];
        const scalar mag = std::sqrt(Sf[f].x*Sf[f].x + Sf[f].y*Sf[f].y + Sf[f].z*Sf[f].z);
        if (mag < 1e-300) continue;
        const scalar nfDotAx = (Sf[f].x*d.axis.x + Sf[f].y*d.axis.y + Sf[f].z*d.axis.z) / mag;
        if (oa != -1 && na == -1)
        {
            if (nfDotAx > 0.8) area[oa] += mag;
        }
        else if (oa == -1 && na != -1)
        {
            if (-nfDotAx > 0.8) area[na] += mag;
        }
    }
    d.cells.copyFrom(p.cells);
    d.e2x.copyFrom(e2x);
    d.e2y.copyFrom(e2y);
    d.e2z.copyFrom(e2z);
    d.radius.copyFrom(rad);
    d.area.copyFrom(area);
    d.twist.copyFrom(tw);
    d.chord.copyFrom(ch);
    d.pAlpha.copyFrom(p.pAlpha);
    d.pCd.copyFrom(p.pCd);
    d.pCl.copyFrom(p.pCl);
    d.nProf = (int)p.pAlpha.size();
    return d;
}

void deviceRotorForce(
    const DeviceRotorDisk& rd,
    int nCells,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& fx,
    DeviceBuffer<scalar>& fy,
    DeviceBuffer<scalar>& fz)
{
    if (!rd.active || rd.n == 0) return;
    fx.resize(nCells);
    fy.resize(nCells);
    fz.resize(nCells);
    cudaCheck(cudaMemsetAsync(fx.data(), 0, nCells*sizeof(scalar), cudaStreamPerThread), "rotor fx zero");
    cudaCheck(cudaMemsetAsync(fy.data(), 0, nCells*sizeof(scalar), cudaStreamPerThread), "rotor fy zero");
    cudaCheck(cudaMemsetAsync(fz.data(), 0, nCells*sizeof(scalar), cudaStreamPerThread), "rotor fz zero");
    {
        const int n = rd.n; const label* cells = rd.cells.data();
        const vector axis = rd.axis; const scalar omega = rd.omega, nBlades = rd.nBlades, tipEffect = rd.tipEffect, rMax = rd.rMax;
        const scalar theta0 = rd.theta0; const int localInflow = rd.localInflow?1:0; const vector inletVel = rd.inletVel;
        const scalar *e2xd=rd.e2x.data(), *e2yd=rd.e2y.data(), *e2zd=rd.e2z.data(), *radiusd=rd.radius.data();
        const scalar *aread=rd.area.data(), *twistd=rd.twist.data(), *chordd=rd.chord.data();
        const scalar *pAd=rd.pAlpha.data(), *pCdd=rd.pCd.data(), *pCld=rd.pCl.data(); const int nProf = rd.nProf;
        const scalar *Uxd=Ux.data(), *Uyd=Uy.data(), *Uzd=Uz.data();
        scalar *fxd=fx.data(), *fyd=fy.data(), *fzd=fz.data();
        pcudaParallelFor(nBlocks(rd.n), TPB, [=] __device__ () {
            rotorForceKernel(n, cells, axis, omega, nBlades, tipEffect, rMax, theta0, localInflow, inletVel,
                             e2xd, e2yd, e2zd, radiusd, aread, twistd, chordd, pAd, pCdd, pCld, nProf,
                             Uxd, Uyd, Uzd, fxd, fyd, fzd);
        });
    }
    cudaCheck(cudaGetLastError(), "rotorForce");
}

} // namespace brae
