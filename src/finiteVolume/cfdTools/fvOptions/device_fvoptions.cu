// cf device DarcyForchheimer porosity. See device_fvoptions.cuh. Mirrors OF porosityModels::DarcyForchheimer::apply
// (incompressible: mu=nu, rho=1), implicit isotropic resistance into the diagonal + explicit anisotropic remainder.
#include "device_fvoptions.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__device__ __forceinline__
void cdComponents(
    scalar nu,
    scalar magU,
    scalar dx,
    scalar dy,
    scalar dz,
    scalar fx,
    scalar fy,
    scalar fz,
    scalar& cx,
    scalar& cy,
    scalar& cz)
{
    cx = nu*dx + magU*0.5*fx;
    cy = nu*dy + magU*0.5*fy;
    cz = nu*dz + magU*0.5*fz;
}


// fixedCoeff. OF (fixedCoeff::apply):
//     Cd    = rho*(alpha + beta*mag(U))          [full tensors]
//     isoCd = tr(Cd)
//     Udiag   += V*isoCd
//     Usource -= V*((Cd - I*isoCd) & U)
// The diagonal takes the ISOTROPIC part and the source takes the DEVIATORIC remainder, which is what makes
// an anisotropic resistance implicit in its trace and explicit in the rest.
__device__ __forceinline__
void fixedCd(const scalar* a, const scalar* b, scalar rho, scalar magU, scalar* Cd)
{
    for (int k = 0; k < 9; ++k) Cd[k] = rho*(a[k] + b[k]*magU);
}

__device__
void porFixedDiagKernel(
    int n,
    const label* __restrict__ cells,
    scalar rho,
    const scalar* __restrict__ a,      // 9, row-major
    const scalar* __restrict__ b,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ diag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    scalar Cd[9]; fixedCd(a, b, rho, magU, Cd);
    diag[c] += V[c]*(Cd[0] + Cd[4] + Cd[8]);                       // += V*tr(Cd)
}

__device__
void porFixedSrcKernel(
    int n,
    const label* __restrict__ cells,
    int comp,
    scalar rho,
    const scalar* __restrict__ a,
    const scalar* __restrict__ b,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ src)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    scalar Cd[9]; fixedCd(a, b, rho, magU, Cd);
    const scalar isoCd = Cd[0] + Cd[4] + Cd[8];
    Cd[0] -= isoCd; Cd[4] -= isoCd; Cd[8] -= isoCd;                // Cd - I*isoCd
    const scalar u[3] = {Ux[c], Uy[c], Uz[c]};
    const scalar row = Cd[3*comp+0]*u[0] + Cd[3*comp+1]*u[1] + Cd[3*comp+2]*u[2];
    src[c] -= V[c]*row;                                            // -= V*((Cd - I*isoCd) & U)
}


__device__
void porDiagKernel(
    int n,
    const label* __restrict__ cells,
    scalar nu,
    scalar dx,
    scalar dy,
    scalar dz,
    scalar fx,
    scalar fy,
    scalar fz,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ diag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    scalar cx, cy, cz;
    cdComponents(nu, magU, dx,dy,dz, fx,fy,fz, cx,cy,cz);
    diag[c] += V[c]*(cx + cy + cz);                                    // += V*isoCd  (cellZone cells unique -> no atomic)
}


__device__
void porSrcKernel(
    int n,
    const label* __restrict__ cells,
    int comp,
    scalar nu,
    scalar dx,
    scalar dy,
    scalar dz,
    scalar fx,
    scalar fy,
    scalar fz,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ src)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    scalar cx, cy, cz;
    cdComponents(nu, magU, dx,dy,dz, fx,fy,fz, cx,cy,cz);
    const scalar iso = cx + cy + cz;
    const scalar ccomp = (comp==0)?cx:(comp==1)?cy:cz;
    const scalar Uc    = ((comp==0)?Ux:(comp==1)?Uy:Uz)[c];
    src[c] += V[c]*(iso - ccomp)*Uc;                                   // -= V*((Cd-I*iso).U)[comp]
}
} // namespace


void deviceFvoPorosityDiag(
    const DevicePorosity& por,
    scalar nu,
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& diag)
{
    const int n = static_cast<int>(por.cells.size());
    if (!por.active || !n) return;
    const label* cellsd = por.cells.data();
    const scalar *Vd = V.data(), *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data();
    scalar* diagd = diag.data();
    if (por.fixed)
    {
        DeviceBuffer<scalar> a, b; a.copyFrom(std::vector<scalar>(por.fa, por.fa+9)); b.copyFrom(std::vector<scalar>(por.fb, por.fb+9));
        const scalar rho = por.rhoRef; const scalar *ad = a.data(), *bd = b.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            porFixedDiagKernel(n, cellsd, rho, ad, bd, Vd, Uxd, Uyd, Uzd, diagd);
        });
        cudaCheck(cudaGetLastError(), "porosityFixedDiag");
        return;
    }
    {
        const scalar dx=por.d.x,dy=por.d.y,dz=por.d.z,fx=por.f.x,fy=por.f.y,fz=por.f.z;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            porDiagKernel(n, cellsd, nu, dx, dy, dz, fx, fy, fz, Vd, Uxd, Uyd, Uzd, diagd);
        });
    }
    cudaCheck(cudaGetLastError(), "porDiag");
}


void deviceFvoPorositySource(
    const DevicePorosity& por,
    int comp,
    scalar nu,
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& src)
{
    const int n = static_cast<int>(por.cells.size());
    if (!por.active || !n) return;
    const label* cellsd = por.cells.data();
    const scalar *Vd = V.data(), *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data();
    scalar* srcd = src.data();
    if (por.fixed)
    {
        DeviceBuffer<scalar> a, b; a.copyFrom(std::vector<scalar>(por.fa, por.fa+9)); b.copyFrom(std::vector<scalar>(por.fb, por.fb+9));
        const scalar rho = por.rhoRef; const scalar *ad = a.data(), *bd = b.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            porFixedSrcKernel(n, cellsd, comp, rho, ad, bd, Vd, Uxd, Uyd, Uzd, srcd);
        });
        cudaCheck(cudaGetLastError(), "porosityFixedSrc");
        return;
    }
    {
        const scalar dx=por.d.x,dy=por.d.y,dz=por.d.z,fx=por.f.x,fy=por.f.y,fz=por.f.z;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            porSrcKernel(n, cellsd, comp, nu, dx, dy, dz, fx, fy, fz, Vd, Uxd, Uyd, Uzd, srcd);
        });
    }
    cudaCheck(cudaGetLastError(), "porSrc");
}


namespace {
__device__
void limitUKernel(
    int n,
    const label* __restrict__ cells,
    scalar maxSqrU,
    scalar* __restrict__ Ux,
    scalar* __restrict__ Uy,
    scalar* __restrict__ Uz)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magSqr = Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c];
    if (magSqr > maxSqrU)
    {
        const scalar s = sqrt(maxSqrU/magSqr);
        Ux[c]*=s;
        Uy[c]*=s;
        Uz[c]*=s;
    }
}
} // namespace


// he clamp on a cell selection. Same shape as limitUKernel: gather through the selection list.
__device__
void limitEnergyKernel(int n, const label* __restrict__ cells, scalar heMin, scalar heMax,
                       scalar* __restrict__ he)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label c = cells[i];
    if      (he[c] < heMin) he[c] = heMin;
    else if (he[c] > heMax) he[c] = heMax;
}

void deviceFvoLimitEnergy(
    const DeviceBuffer<label>& cells,
    scalar heMin,
    scalar heMax,
    DeviceBuffer<scalar>& he)
{
    const int n = static_cast<int>(cells.size());
    if (!n) return;
    const label* cellsd = cells.data(); scalar* hed = he.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { limitEnergyKernel(n, cellsd, heMin, heMax, hed); });
    cudaCheck(cudaGetLastError(), "limitTemperature");
}

// Boundary half: every face whose patch does not fix a value. OF's test is fvPatchField::fixesValue(); on
// the device that is bcType == 1 (brae: 0 extrapolated, 1 fixedValue, 2 calculated), and an inletOutlet
// face has already been resolved to 0|1 for this iteration, so the same test covers it.
__device__
void limitEnergyBndKernel(int n, const label* __restrict__ bcType, scalar heMin, scalar heMax,
                          scalar* __restrict__ heBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (bcType[i] == 1) return;                     // fixesValue() -> OF leaves it alone
    if      (heBnd[i] < heMin) heBnd[i] = heMin;
    else if (heBnd[i] > heMax) heBnd[i] = heMax;
}

void deviceFvoLimitEnergyBoundary(
    const DeviceBoundary& dbHe,
    scalar heMin,
    scalar heMax,
    DeviceBuffer<scalar>& heBnd)
{
    const int n = static_cast<int>(heBnd.size());
    if (!n || dbHe.n != n) return;
    const label* bcTyped = dbHe.bcType.data(); scalar* heBndd = heBnd.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { limitEnergyBndKernel(n, bcTyped, heMin, heMax, heBndd); });
    cudaCheck(cudaGetLastError(), "limitTemperatureBnd");
}

void deviceFvoLimitVelocity(
    const DeviceBuffer<label>& cells,
    scalar maxU,
    DeviceBuffer<scalar>& Ux,
    DeviceBuffer<scalar>& Uy,
    DeviceBuffer<scalar>& Uz)
{
    const int n = static_cast<int>(cells.size());
    if (!n) return;
    const label* cellsd = cells.data(); const scalar maxSqrU = maxU*maxU;
    scalar *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { limitUKernel(n, cellsd, maxSqrU, Uxd, Uyd, Uzd); });
    cudaCheck(cudaGetLastError(), "limitVelocity");
}


// velocityDampingConstraint: diag[c] += C*V[c]^(2/3)*(|U|-UMax) where |U| > UMax.
__device__
void velDampKernel(
    int n,
    const label* __restrict__ cells,
    scalar UMax,
    scalar C,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ diag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    if (magU > UMax)
    {
        const scalar s = cbrt(V[c]);
        diag[c] += C*s*s*(magU - UMax);   // s*s = V^(2/3)
    }
}


void deviceFvoVelocityDamping(
    const DeviceBuffer<label>& cells,
    scalar UMax,
    scalar C,
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& diag)
{
    const int n = static_cast<int>(cells.size());
    if (!n) return;
    const label* cellsd = cells.data();
    const scalar *Vd = V.data(), *Uxd = Ux.data(), *Uyd = Uy.data(), *Uzd = Uz.data();
    scalar* diagd = diag.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { velDampKernel(n, cellsd, UMax, C, Vd, Uxd, Uyd, Uzd, diagd); });
    cudaCheck(cudaGetLastError(), "velocityDampingConstraint");
}

} // namespace brae
