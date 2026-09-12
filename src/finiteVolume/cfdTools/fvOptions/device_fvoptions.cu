// cf device DarcyForchheimer porosity. See device_fvoptions.cuh. Mirrors OF porosityModels::DarcyForchheimer::apply
// (incompressible: mu=nu, rho=1), implicit isotropic resistance into the diagonal + explicit anisotropic remainder.
#include "device_fvoptions.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {

// fvMatrix::setValues -- what OpenFOAM's fvOptions CONSTRAINTS do, and what a wall function's
// manipulateMatrix does. NOT the same as overwriting the field after the solve: setValues also
// TRANSFERS the pinned value into every neighbour's source and then zeroes the coefficient it
// transferred through, so the constraint reaches the cells around it exactly once.
//
// Hoisted here from the kEpsilon closure, which had the only copy. The energy equation needs the same
// four kernels for fixedTemperatureConstraint, and two copies of a matrix manipulation this delicate
// is how the two constraints stop agreeing about what OpenFOAM does.
// A GATHER, and it has to be one (queue items 59/69). This ran as a scatter -- one thread per PINNED
// cell, atomicAdd into each neighbour's source -- so a cell next to two or more pinned cells summed its
// contributions in whatever order the blocks happened to finish, and the last bit of its source moved
// from run to run. That is where the compressible mirror's nondeterminism entered: on sbMatched two
// identical runs agreed through every stage of iteration 1 and diverged first in the epsilon SOURCE at
// iteration 2, which is this kernel under the epsilon wall constraint (every wall-adjacent cell is
// pinned, so cells beside two of them are common). Same terms, one per face, now summed by the
// RECEIVING cell in face order -- the losort side then the owner side, the order gsCellUpdate and
// gaussGrad already use -- so the result is fixed and the run is reproducible.
__global__ void svGatherKernel(
    int           nC,
    const label*  mask,
    const scalar* value,
    const label*  ownerStart,
    const label*  losort,
    const label*  losortStart,
    const label*  own,
    const label*  nei,
    const scalar* upper,
    const scalar* lower,
    scalar*       source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // c RECEIVES from every face whose other cell is pinned. The face's coefficient is the one the
    // scatter used: `lower` when the pinned cell owns the face, `upper` when it is the neighbour.
    scalar acc = source[c];
    for (label s = losortStart[c]; s < losortStart[c + 1]; ++s)       // c is the neighbour of face f
    {
        const label f = losort[s];
        const label o = own[f];
        if (mask[o]) acc -= lower[f] * value[o];
    }
    for (label f = ownerStart[c]; f < ownerStart[c + 1]; ++f)         // c is the owner of face f
    {
        const label n = nei[f];
        if (mask[n]) acc -= upper[f] * value[n];
    }
    source[c] = acc;
}


__global__ void svZeroFaceKernel(
    int          nIf,
    const label* own,
    const label* nei,
    const label* mask,
    scalar*      upper,
    scalar*      lower)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    if (mask[own[f]] || mask[nei[f]])
    {
        upper[f] = scalar(0.0);
        lower[f] = scalar(0.0);
    }
}


// Every BOUNDARY face of a constrained cell loses its coefficients too -- including a face on a patch
// that has nothing to do with the wall function. Without this an outlet div(phi) coefficient on a
// wall/outlet corner cell is folded into the diagonal at solve time and pulls the pinned cell off its
// value.
__global__ void svBndKernel(
    int          nB,
    const label* bndCell,
    const label* mask,
    scalar*      iC,
    scalar*      bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    if (!mask[bndCell[i]]) return;
    iC[i] = scalar(0.0);
    bC[i] = scalar(0.0);
}


// ...and only THEN psi and the source, so that adjacent constrained cells cannot corrupt each other's.
__global__ void svCellKernel(
    int           nC,
    const label*  mask,
    const scalar* value,
    const scalar* diag,
    scalar*       psi,
    scalar*       source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (!mask[c]) return;
    psi[c]    = value[c];
    source[c] = value[c] * diag[c];
}

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

__global__
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

__global__
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


__global__
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


__global__
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
    if (por.fixed)
    {
        DeviceBuffer<scalar> a, b; a.copyFrom(std::vector<scalar>(por.fa, por.fa+9)); b.copyFrom(std::vector<scalar>(por.fb, por.fb+9));
        porFixedDiagKernel<<<nBlocks(n), TPB>>>(n, por.cells.data(), por.rhoRef, a.data(), b.data(),
                                                V.data(), Ux.data(), Uy.data(), Uz.data(), diag.data());
        cudaCheck(cudaGetLastError(), "porosityFixedDiag");
        return;
    }
    porDiagKernel<<<nBlocks(n), TPB>>>(n, por.cells.data(), nu, por.d.x,por.d.y,por.d.z, por.f.x,por.f.y,por.f.z,
                                       V.data(), Ux.data(), Uy.data(), Uz.data(), diag.data());
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
    if (por.fixed)
    {
        DeviceBuffer<scalar> a, b; a.copyFrom(std::vector<scalar>(por.fa, por.fa+9)); b.copyFrom(std::vector<scalar>(por.fb, por.fb+9));
        porFixedSrcKernel<<<nBlocks(n), TPB>>>(n, por.cells.data(), comp, por.rhoRef, a.data(), b.data(),
                                               V.data(), Ux.data(), Uy.data(), Uz.data(), src.data());
        cudaCheck(cudaGetLastError(), "porosityFixedSrc");
        return;
    }
    porSrcKernel<<<nBlocks(n), TPB>>>(n, por.cells.data(), comp, nu, por.d.x,por.d.y,por.d.z, por.f.x,por.f.y,por.f.z,
                                      V.data(), Ux.data(), Uy.data(), Uz.data(), src.data());
    cudaCheck(cudaGetLastError(), "porSrc");
}


namespace {
__global__
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
__global__
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
    limitEnergyKernel<<<nBlocks(n), TPB>>>(n, cells.data(), heMin, heMax, he.data());
    cudaCheck(cudaGetLastError(), "limitTemperature");
}

// Boundary half: every face whose patch does not fix a value. OF's test is fvPatchField::fixesValue(); on
// the device that is bcType == 1 (brae: 0 extrapolated, 1 fixedValue, 2 calculated), and an inletOutlet
// face has already been resolved to 0|1 for this iteration, so the same test covers it.
__global__
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
    limitEnergyBndKernel<<<nBlocks(n), TPB>>>(n, dbHe.bcType.data(), heMin, heMax, heBnd.data());
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
    limitUKernel<<<nBlocks(n), TPB>>>(n, cells.data(), maxU*maxU, Ux.data(), Uy.data(), Uz.data());
    cudaCheck(cudaGetLastError(), "limitVelocity");
}


// velocityDampingConstraint: diag[c] += C*V[c]^(2/3)*(|U|-UMax) where |U| > UMax.
__global__
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
    velDampKernel<<<nBlocks(n), TPB>>>(n, cells.data(), UMax, C, V.data(), Ux.data(), Uy.data(), Uz.data(), diag.data());
    cudaCheck(cudaGetLastError(), "velocityDampingConstraint");
}

// One constraint, applied to an assembled matrix in OpenFOAM's order (relax -> constrain). `mask` is
// per CELL (non-zero = pinned) and `value` per cell; both are the caller's, because what is pinned and
// to what is the option's business, not the matrix's.
void deviceSetValues(
    const DeviceMesh&           dm,
    const DeviceBuffer<label>&  mask,
    const DeviceBuffer<scalar>& value,
    DeviceBuffer<scalar>&       diag,
    DeviceBuffer<scalar>&       upper,
    DeviceBuffer<scalar>&       lower,
    DeviceBuffer<scalar>&       source,
    DeviceBuffer<scalar>&       internalCoeffs,
    DeviceBuffer<scalar>&       boundaryCoeffs,
    DeviceBuffer<scalar>&       psi)
{
    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nB  = dm.nBndFaces;
    if (nC == 0 || static_cast<int>(mask.size()) != nC) return;

    svGatherKernel<<<nBlocks(nC), TPB>>>(nC, mask.data(), value.data(), dm.ownerStart.data(),
                                         dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
                                         dm.nei.data(), upper.data(), lower.data(), source.data());
    cudaCheck(cudaGetLastError(), "fvOptions setValues gather");
    if (nIf > 0)
    {
        svZeroFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), mask.data(),
                                                upper.data(), lower.data());
        cudaCheck(cudaGetLastError(), "fvOptions setValues zero faces");
    }
    if (nB > 0 && internalCoeffs.size() && boundaryCoeffs.size())
    {
        svBndKernel<<<nBlocks(nB), TPB>>>(nB, dm.bndCell.data(), mask.data(),
                                          internalCoeffs.data(), boundaryCoeffs.data());
        cudaCheck(cudaGetLastError(), "fvOptions setValues zero boundary");
    }
    svCellKernel<<<nBlocks(nC), TPB>>>(nC, mask.data(), value.data(), diag.data(), psi.data(),
                                       source.data());
    cudaCheck(cudaGetLastError(), "fvOptions setValues cells");
}

} // namespace brae
