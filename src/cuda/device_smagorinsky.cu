// Smagorinsky LES sub-grid model: the ALGEBRAIC eddy viscosity nut = Ck*delta*sqrt(k_sgs), transcribed from OF's
// Smagorinsky<BasicTurbulenceModel>::k(gradU) + correctNut(). Pure LES -> no scalar transport (no k/epsilon/omega/
// nuTilda equation); nut is a closed-form function of the local strain and cell size, recomputed each step from U.
//   D = symm(gradU);  a = Ce/delta;  b = (2/3)tr(D);  c = 2*Ck*delta*(dev(D) && D);
//   sqrt(k_sgs) = (-b + sqrt(b^2 + 4ac))/(2a);  nut = Ck*delta*sqrt(k_sgs),  delta = cbrt(V) (cubeRootVol).
// dev(D) && D = D:D - (1/3)tr(D)^2 = |dev(D)|^2 >= 0, so the discriminant b^2+4ac >= 0 and sqrt(k_sgs) >= 0.
// Incompressible (tr(D)=div(U)->0): nut = Ck*sqrt(2*Ck/Ce)*delta^2*sqrt(S:S), the classic Cs~0.168 Smagorinsky.
#include "device_smagorinsky.cuh"
#include "device_scalar_transport.cuh"  // nBlocks/TPB (shared launch geometry)
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {

__device__
void smagorinskyNutKernel(
    int nC, const scalar* __restrict__ gradU, const scalar* __restrict__ V,
    SmagorinskyCoeffs co, const scalar* __restrict__ dOpt, scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];
    const scalar delta = dOpt ? dOpt[c] : cbrt(V[c]);             // LES filter width: the case's `delta`
    const scalar trD = t[0] + t[4] + t[8];                        // tr(symm gradU) = div(U)
    scalar DD = 0;                                                // D:D = magSqr(symm(gradU))
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar sab = scalar(0.5) * (t[i*3+j] + t[j*3+i]);
            DD += sab * sab;
        }
    const scalar devDD = fmax(DD - trD*trD/scalar(3), scalar(0));  // dev(D) && D = |dev(D)|^2 (>= 0)
    // OF Smagorinsky<>::k(): sqrt(k_sgs) = (-b + sqrt(b^2 + 4ac))/(2a)
    const scalar a = co.Ce / fmax(delta, scalar(1e-300));
    const scalar b = (scalar(2)/scalar(3)) * trD;
    const scalar cc = scalar(2) * co.Ck * delta * devDD;
    const scalar sqrtk = (-b + sqrt(fmax(b*b + scalar(4)*a*cc, scalar(0)))) / (scalar(2)*a);
    nut[c] = co.Ck * delta * fmax(sqrtk, scalar(0));              // nut = Ck*delta*sqrt(k_sgs)
}

} // namespace

namespace {

// OF LESModels::WALE::k(gradU), transcribed literally so the SMALL sits where OF puts it:
//     Sd        = devSymm(gradU & gradU)                     [tensor product, then symm, then deviatoric]
//     magSqrSd  = magSqr(Sd)
//     k         = sqr(sqr(Cw)*delta/Ck) * pow3(magSqrSd)
//                 / ( sqr( pow(magSqr(symm(gradU)), 5/2) + pow(magSqrSd, 5/4) ) + SMALL )
//     nut       = Ck*delta*sqrt(k)
// gradU is packed q = 3i + j = d(U_j)/d(x_i), which is OF's own tensor layout, so (A & B)[i][j] is
// sum_k A[i][k]*B[k][j] on the same indices.
__device__
void waleNutKernel(
    int nC, const scalar* __restrict__ gradU, const scalar* __restrict__ V,
    WaleCoeffs co, const scalar* __restrict__ dOpt, scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];
    const scalar delta = dOpt ? dOpt[c] : cbrt(V[c]);

    scalar gg[9];                                                  // gradU & gradU
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            scalar sum = 0;
            for (int k = 0; k < 3; ++k) sum += t[i*3+k] * t[k*3+j];
            gg[i*3+j] = sum;
        }
    const scalar trGG = gg[0] + gg[4] + gg[8];
    scalar magSqrSd = 0, magSqrS = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            // devSymm(gg) = symm(gg) - (1/3)tr(gg) I
            scalar sd = scalar(0.5)*(gg[i*3+j] + gg[j*3+i]);
            if (i == j) sd -= trGG/scalar(3);
            magSqrSd += sd*sd;
            const scalar s = scalar(0.5)*(t[i*3+j] + t[j*3+i]);    // symm(gradU)
            magSqrS += s*s;
        }
    const scalar A   = co.Cw*co.Cw*delta/co.Ck;
    const scalar den = pow(magSqrS, scalar(2.5)) + pow(magSqrSd, scalar(1.25));
    const scalar k   = A*A * (magSqrSd*magSqrSd*magSqrSd) / (den*den + scalar(1e-15));
    nut[c] = co.Ck * delta * sqrt(fmax(k, scalar(0)));
}

} // namespace


void deviceWaleNut(int nC, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& V,
                   const WaleCoeffs& co, DeviceBuffer<scalar>& nut, const DeviceBuffer<scalar>* delta)
{
    nut.resize(nC);
    if (nC == 0) return;
    const scalar* gradUd = gradU.data();
    const scalar* Vd = V.data();
    const scalar* deltad = (delta && delta->size()) ? delta->data() : nullptr;
    scalar* nutd = nut.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { waleNutKernel(nC, gradUd, Vd, co, deltad, nutd); });
    cudaCheck(cudaGetLastError(), "deviceWaleNut");
}


void deviceSmagorinskyNut(int nC, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& V,
                          const SmagorinskyCoeffs& co, DeviceBuffer<scalar>& nut,
                          const DeviceBuffer<scalar>* delta)
{
    nut.resize(nC);
    if (nC == 0) return;
    const scalar* gradUd = gradU.data();
    const scalar* Vd = V.data();
    const scalar* deltad = (delta && delta->size()) ? delta->data() : nullptr;
    scalar* nutd = nut.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { smagorinskyNutKernel(nC, gradUd, Vd, co, deltad, nutd); });
    cudaCheck(cudaGetLastError(), "deviceSmagorinskyNut");
}

} // namespace brae
