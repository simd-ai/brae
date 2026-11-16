// functionObjects::forces' face integration on the device: what it computes and in which order is said in forces_device.cuh.
#include "forces_device.cuh"

#include "device_buffer.cuh"

#include <stdexcept>
#include <string>

namespace brae
{

namespace
{

constexpr int kForceThreads = 256;

__device__ inline void cross3(
    const double a[3],
    const double b[3],
    double out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

// one block a selected patch: each thread its contiguous run of the patch's faces, in order, then the block's fixed tree
__global__
void forcePatchSumsKernel(
    const int* __restrict__ patchOffset,
    const label* __restrict__ bnd,
    const scalar* __restrict__ Sf,
    const scalar* __restrict__ Cf,
    const label* __restrict__ faceCellB,
    const scalar* __restrict__ dcB,
    const scalar* __restrict__ pB,
    const scalar* __restrict__ UBx,
    const scalar* __restrict__ UBy,
    const scalar* __restrict__ UBz,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ gradU,
    long long nCells,
    const scalar* __restrict__ nuEffB,
    const scalar* __restrict__ rhoB,
    scalar rhoP,
    scalar rhoV,
    scalar pRef,
    int correctGrad,
    scalar ox,
    scalar oy,
    scalar oz,
    double* __restrict__ out)
{
    __shared__ double part[12][kForceThreads];
    const int patch = blockIdx.x;
    const int begin = patchOffset[patch];
    const int end = patchOffset[patch + 1];
    const int n = end - begin;
    const int per = (n + kForceThreads - 1) / kForceThreads;
    const int first = begin + threadIdx.x * per;
    const int last = min(end, first + per);
    double s[12];
    for (int k = 0; k < 12; ++k) s[k] = 0.0;
    for (int f = first; f < last; ++f)
    {
        const label b = bnd[f];
        const label c = faceCellB[b];
        const double dc = dcB[b];
        const double S[3] = {Sf[3 * f], Sf[3 * f + 1], Sf[3 * f + 2]};
        const double magS = sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
        const double nh[3] = {S[0] / magS, S[1] / magS, S[2] / magS};
        // fP = rhoP*Sf*(p_b - pRef), in forces.C's order: (rhoRef*Sf)*(p_b - pRef)
        const double dp = pB[b] - pRef;
        const double fP[3] = {(rhoP * S[0]) * dp, (rhoP * S[1]) * dp, (rhoP * S[2]) * dp};
        // gradU_b = gradU_c + n*(snGrad - (n & gradU_c)), snGrad = (U_b - U_c)*deltaCoeffs
        const double sn[3] = {dc * (UBx[b] - Ux[c]), dc * (UBy[b] - Uy[c]), dc * (UBz[b] - Uz[c])};
        double G[9];
        for (int k = 0; k < 9; ++k) G[k] = gradU[static_cast<long long>(k) * nCells + c];
        double nG[3];
        for (int j = 0; j < 3; ++j) nG[j] = nh[0] * G[j] + nh[1] * G[3 + j] + nh[2] * G[6 + j];
        if (correctGrad)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) G[3 * i + j] += nh[i] * (sn[j] - nG[j]);
        // devTwoSymm(G) = G + G^T - (2/3) tr(G) I; fV = Sf & (-rho*nuEff*devTwoSymm(G))
        const double tr = (2.0 / 3.0) * (G[0] + G[4] + G[8]);
        const double rho = rhoB ? rhoB[b] : rhoV;
        const double coef = -rho * nuEffB[b];
        double fV[3];
        for (int j = 0; j < 3; ++j)
        {
            double a = 0.0;
            for (int i = 0; i < 3; ++i)
            {
                const double d = G[3 * i + j] + G[3 * j + i] - (i == j ? tr : 0.0);
                a += S[i] * (coef * d);
            }
            fV[j] = a;
        }
        const double Md[3] = {Cf[3 * f] - ox, Cf[3 * f + 1] - oy, Cf[3 * f + 2] - oz};
        double mP[3];
        double mV[3];
        cross3(Md, fP, mP);
        cross3(Md, fV, mV);
        for (int k = 0; k < 3; ++k)
        {
            s[k] += fP[k];
            s[3 + k] += fV[k];
            s[6 + k] += mP[k];
            s[9 + k] += mV[k];
        }
    }
    for (int k = 0; k < 12; ++k) part[k][threadIdx.x] = s[k];
    __syncthreads();
    for (int w = kForceThreads / 2; w > 0; w /= 2)
    {
        if (threadIdx.x < w)
            for (int k = 0; k < 12; ++k) part[k][threadIdx.x] += part[k][threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int k = 0; k < 12; ++k) out[12 * patch + k] = part[k][0];
}

} // namespace

std::vector<double> deviceForcePatchSums(
    const DeviceForceFaces& faces,
    const DeviceForceFields& fields)
{
    const int nPatches = static_cast<int>(faces.patchOffset.size()) - 1;
    std::vector<double> out(static_cast<std::size_t>(12 * (nPatches > 0 ? nPatches : 0)), 0.0);
    if (nPatches <= 0 || faces.nFaces == 0) return out;
    if (!fields.faceCellB || !fields.dcB || !fields.pB || !fields.UB[0] || !fields.U[0] || !fields.gradU || !fields.nuEffB || fields.nCells <= 0)
        throw std::logic_error("deviceForcePatchSums: a field the face integration reads is missing");
    DeviceBuffer<int> offset;
    offset.copyFrom(faces.patchOffset);
    DeviceBuffer<double> sums(out.size());
    forcePatchSumsKernel<<<nPatches, kForceThreads, 0, cudaStreamPerThread>>>(
        offset.data(), faces.bnd.data(), faces.Sf.data(), faces.Cf.data(), fields.faceCellB, fields.dcB, fields.pB,
        fields.UB[0], fields.UB[1], fields.UB[2], fields.U[0], fields.U[1], fields.U[2], fields.gradU, static_cast<long long>(fields.nCells), fields.nuEffB, fields.rhoB,
        fields.rhoP, fields.rhoV, fields.pRef, fields.correctGrad ? 1 : 0, fields.origin.x, fields.origin.y, fields.origin.z, sums.data());
    cudaCheck(cudaGetLastError(), "forcePatchSums");
    return sums.host();
}

} // namespace brae
