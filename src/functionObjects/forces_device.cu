// functionObjects::forces' face integration on the device: what it computes and in which order is said in forces_device.cuh.
#include "forces_device.cuh"

#include "device_buffer.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

// THE FACE-PARALLEL PATH (BRAE_FORCES_FACE_PARALLEL=1). Its summation order is forcePatchSumsKernel's: a patch's faces make
// kForceRuns contiguous runs (that kernel's threads), each run summed in face order from 0.0, then the kForceRuns partial sums
// added in that kernel's tree. Changing either changes the bits of every force.
constexpr int kForceRuns = kForceThreads;
constexpr int kForceTerms = 12;
constexpr int kTermThreads = 256;
constexpr int kSumThreads = 256;
// The faces whose terms are held at once: 2^19 faces, 48 MiB. A selection with more is integrated in chunks, each run carrying
// its partial sums from one chunk to the next, so neither the order nor the bits depend on the chunk size.
constexpr int kChunkFaces = 1 << 19;

// cross3 as forcePatchSumsKernel's SASS evaluates it (sm_90, sm_100, sm_121): out_i = fma(a_j, b_k, -(a_k*b_j))
__device__ __forceinline__
void cross3Rn(
    const double a[3],
    const double b[3],
    double out[3])
{
    out[0] = __fma_rn(a[1], b[2], -__dmul_rn(a[2], b[1]));
    out[1] = __fma_rn(a[2], b[0], -__dmul_rn(a[0], b[2]));
    out[2] = __fma_rn(a[0], b[1], -__dmul_rn(a[1], b[0]));
}

// One face's twelve terms: fP (3), fV (3), mP (3), mV (3), each rounded as forcePatchSumsKernel rounds it. That kernel writes
// the arithmetic as plain expressions and leaves the FMA contractions to the compiler, which makes them by context: the same
// expressions in another kernel contract differently on sm_100 and sm_121 (sn - nG unfused, its product hoisted out of the
// correctGrad branch). So every operation here is written as an intrinsic, in the contraction that kernel's SASS shows
// (cuobjdump, nvcc 13.0, the same on sm_90, sm_100 and sm_121):
//     |S|^2 = fma(S2, S2, fma(S0, S0, S1*S1)); n = S/|S| (IEEE sqrt and division)
//     nG_j = fma(n2, G_6+j, fma(n0, G_j, n1*G_3+j))
//     G_3i+j = fma(n_i, fma(dc, U_b,j - U_c,j, -nG_j), G_3i+j)                (sn_j - nG_j fused)
//     d_ii = fma(G00 + G11 + G22, -(2/3), G_ii + G_ii); d_ij = G_ij + G_ji     (tr fused into each diagonal)
//     fV_j = fma(-S2, rn*d_2j, fma(-S1, rn*d_1j, fma(-S0, rn*d_0j, 0))), rn = rho*nuEff, coef = -rn
//     fP_k = (rhoP*S_k)*(p_b - pRef); M = Cf - origin; m = cross3Rn(M, f)
// The sum of a term into its run is a plain DADD there (the terms are rounded before they are added), and the same here.
// tests/test_forces_face_parallel.cu compares the two paths bit for bit.
__device__ __forceinline__
void forceFaceTerms(
    int f,
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
    double t[kForceTerms])
{
    const label b = bnd[f];
    const label c = faceCellB[b];
    const double S[3] = {Sf[3 * f], Sf[3 * f + 1], Sf[3 * f + 2]};
    const double magS = __dsqrt_rn(__fma_rn(S[2], S[2], __fma_rn(S[0], S[0], __dmul_rn(S[1], S[1]))));
    const double nh[3] = {__ddiv_rn(S[0], magS), __ddiv_rn(S[1], magS), __ddiv_rn(S[2], magS)};
    // fP = rhoP*Sf*(p_b - pRef), in forces.C's order: (rhoRef*Sf)*(p_b - pRef)
    const double dp = __dsub_rn(pB[b], pRef);
    const double fP[3] = {__dmul_rn(__dmul_rn(rhoP, S[0]), dp), __dmul_rn(__dmul_rn(rhoP, S[1]), dp), __dmul_rn(__dmul_rn(rhoP, S[2]), dp)};
    // gradU_b = gradU_c + n*(snGrad - (n & gradU_c)), snGrad = (U_b - U_c)*deltaCoeffs
    double G[9];
    for (int k = 0; k < 9; ++k) G[k] = gradU[static_cast<long long>(k) * nCells + c];
    double nG[3];
    for (int j = 0; j < 3; ++j) nG[j] = __fma_rn(nh[2], G[6 + j], __fma_rn(nh[0], G[j], __dmul_rn(nh[1], G[3 + j])));
    if (correctGrad)
    {
        const double dc = dcB[b];
        const double dU[3] = {__dsub_rn(UBx[b], Ux[c]), __dsub_rn(UBy[b], Uy[c]), __dsub_rn(UBz[b], Uz[c])};
        double e[3];
        for (int j = 0; j < 3; ++j) e[j] = __fma_rn(dc, dU[j], -nG[j]);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) G[3 * i + j] = __fma_rn(nh[i], e[j], G[3 * i + j]);
    }
    // devTwoSymm(G) = G + G^T - (2/3) tr(G) I; fV = Sf & (-rho*nuEff*devTwoSymm(G))
    const double diag = __dadd_rn(__dadd_rn(G[0], G[4]), G[8]);
    const double rho = rhoB ? rhoB[b] : rhoV;
    const double rn = __dmul_rn(rho, nuEffB[b]);
    double fV[3];
    for (int j = 0; j < 3; ++j)
    {
        double a = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            const double d = i == j ? __fma_rn(diag, -(2.0 / 3.0), __dadd_rn(G[3 * i + j], G[3 * i + j])) : __dadd_rn(G[3 * i + j], G[3 * j + i]);
            a = __fma_rn(-S[i], __dmul_rn(rn, d), a);
        }
        fV[j] = a;
    }
    const double Md[3] = {__dsub_rn(Cf[3 * f], ox), __dsub_rn(Cf[3 * f + 1], oy), __dsub_rn(Cf[3 * f + 2], oz)};
    double mP[3];
    double mV[3];
    cross3Rn(Md, fP, mP);
    cross3Rn(Md, fV, mV);
    for (int k = 0; k < 3; ++k)
    {
        t[k] = fP[k];
        t[3 + k] = fV[k];
        t[6 + k] = mP[k];
        t[9 + k] = mV[k];
    }
}

// one thread a face of the chunk [chunkBegin, chunkEnd): its twelve terms at terms[12*(f - chunkBegin) + k]
__global__
void forceFaceTermsKernel(
    int chunkBegin,
    int chunkEnd,
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
    double* __restrict__ terms)
{
    const int f = chunkBegin + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (f >= chunkEnd) return;
    double t[kForceTerms];
    forceFaceTerms(f, bnd, Sf, Cf, faceCellB, dcB, pB, UBx, UBy, UBz, Ux, Uy, Uz, gradU, nCells, nuEffB, rhoB, rhoP, rhoV, pRef,
                   correctGrad, ox, oy, oz, t);
    double* out = terms + static_cast<long long>(kForceTerms) * (f - chunkBegin);
    for (int k = 0; k < kForceTerms; ++k) out[k] = t[k];
}

// One thread a (patch, run, term): the run's faces inside the chunk, in face order, added to the run's partial sum. A patch's n
// faces make kForceRuns runs of per = ceil(n/kForceRuns) faces, run r holding [begin + r*per, min(end, begin + (r + 1)*per)).
// The first chunk starts every partial sum at 0.0; a later one continues it from runSums[(12*patch + k)*kForceRuns + r].
// TEST CONTROL reversePatch (>= 0): run 0 of that patch adds its faces in reverse, a deliberately wrong order.
__global__
void forceRunSumsKernel(
    int nPatches,
    const int* __restrict__ patchOffset,
    int chunkBegin,
    int chunkEnd,
    int firstChunk,
    int reversePatch,
    const double* __restrict__ terms,
    double* __restrict__ runSums)
{
    const long long g = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int k = static_cast<int>(g % kForceTerms);
    const long long q = g / kForceTerms;
    const int r = static_cast<int>(q % kForceRuns);
    const int patch = static_cast<int>(q / kForceRuns);
    if (patch >= nPatches) return;
    const int begin = patchOffset[patch];
    const int end = patchOffset[patch + 1];
    const int n = end - begin;
    const int per = (n + kForceRuns - 1) / kForceRuns;
    const int first = begin + r * per;
    const int last = min(end, first + per);
    const int lo = max(first, chunkBegin);
    const int hi = min(last, chunkEnd);
    double* slot = runSums + (static_cast<long long>(kForceTerms) * patch + k) * kForceRuns + r;
    double s = firstChunk ? 0.0 : *slot;
    const double* x = terms + k;
    if (patch == reversePatch && r == 0)
    {
        for (int f = hi - 1; f >= lo; --f) s += x[static_cast<long long>(kForceTerms) * (f - chunkBegin)];
        *slot = s;
        return;
    }
    // the loads of kAhead faces are issued before their additions, which stay one after another in face order
    constexpr int kAhead = 16;
    int f = lo;
    for (; f + kAhead <= hi; f += kAhead)
    {
        double v[kAhead];
#pragma unroll
        for (int u = 0; u < kAhead; ++u) v[u] = x[static_cast<long long>(kForceTerms) * (f + u - chunkBegin)];
#pragma unroll
        for (int u = 0; u < kAhead; ++u) s += v[u];
    }
    for (; f < hi; ++f) s += x[static_cast<long long>(kForceTerms) * (f - chunkBegin)];
    *slot = s;
}

// one block a patch: its kForceRuns partial sums added in forcePatchSumsKernel's fixed tree
__global__
void forceRunTreeKernel(
    const double* __restrict__ runSums,
    double* __restrict__ out)
{
    __shared__ double part[kForceTerms][kForceRuns];
    const int patch = blockIdx.x;
    for (int k = 0; k < kForceTerms; ++k) part[k][threadIdx.x] = runSums[(static_cast<long long>(kForceTerms) * patch + k) * kForceRuns + threadIdx.x];
    __syncthreads();
    for (int w = kForceRuns / 2; w > 0; w /= 2)
    {
        if (threadIdx.x < w)
            for (int k = 0; k < kForceTerms; ++k) part[k][threadIdx.x] += part[k][threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int k = 0; k < kForceTerms; ++k) out[kForceTerms * patch + k] = part[k][0];
}

// THE DEFAULT: one block a patch. A call uploads the offsets and takes the sums' buffer from the pool
std::vector<double> patchBlockSums(
    const DeviceForceFaces& faces,
    const DeviceForceFields& fields,
    int nPatches)
{
    DeviceBuffer<int> offset;
    offset.copyFrom(faces.patchOffset);
    DeviceBuffer<double> sums(static_cast<std::size_t>(12 * nPatches));
    forcePatchSumsKernel<<<nPatches, kForceThreads, 0, cudaStreamPerThread>>>(
        offset.data(), faces.bnd.data(), faces.Sf.data(), faces.Cf.data(), fields.faceCellB, fields.dcB, fields.pB,
        fields.UB[0], fields.UB[1], fields.UB[2], fields.U[0], fields.U[1], fields.U[2], fields.gradU, static_cast<long long>(fields.nCells), fields.nuEffB, fields.rhoB,
        fields.rhoP, fields.rhoV, fields.pRef, fields.correctGrad ? 1 : 0, fields.origin.x, fields.origin.y, fields.origin.z, sums.data());
    cudaCheck(cudaGetLastError(), "forcePatchSums");
    return sums.host();
}

// THE FACE-PARALLEL PATH: the terms of a chunk of faces, the runs' partial sums carried across the chunks, then the tree
std::vector<double> faceParallelSums(
    const DeviceForceFaces& faces,
    const DeviceForceFields& fields,
    int nPatches)
{
    std::vector<double> out(static_cast<std::size_t>(kForceTerms * nPatches), 0.0);
    // the selection's device scratch, made at the first call and kept: the patch offsets (uploaded again only when they change),
    // the chunk's terms, the runs' partial sums and the patches' sums. A step allocates nothing and uploads nothing
    if (faces.patchOffsetOnDevice != faces.patchOffset)
    {
        faces.dPatchOffset.copyFrom(faces.patchOffset);
        faces.patchOffsetOnDevice = faces.patchOffset;
    }
    const int chunk = std::min(faces.nFaces, kChunkFaces);
    faces.terms.resize(static_cast<std::size_t>(kForceTerms) * static_cast<std::size_t>(chunk));
    faces.runSums.resize(static_cast<std::size_t>(kForceTerms) * kForceRuns * static_cast<std::size_t>(nPatches));
    faces.sums.resize(out.size());
    // TEST CONTROL: the patch with the most faces, so that its first run is long enough to round differently in reverse
    int reversePatch = -1;
    if (fields.controlReverseRun)
    {
        int most = 0;
        for (int p = 0; p < nPatches; ++p)
        {
            const int n = faces.patchOffset[p + 1] - faces.patchOffset[p];
            if (n <= most) continue;
            most = n;
            reversePatch = p;
        }
    }
    const long long sumThreads = static_cast<long long>(nPatches) * kForceRuns * kForceTerms;
    const int sumBlocks = static_cast<int>((sumThreads + kSumThreads - 1) / kSumThreads);
    for (int c0 = 0; c0 < faces.nFaces; c0 += chunk)
    {
        const int c1 = std::min(faces.nFaces, c0 + chunk);
        const int termBlocks = (c1 - c0 + kTermThreads - 1) / kTermThreads;
        forceFaceTermsKernel<<<termBlocks, kTermThreads, 0, cudaStreamPerThread>>>(
            c0, c1, faces.bnd.data(), faces.Sf.data(), faces.Cf.data(), fields.faceCellB, fields.dcB, fields.pB, fields.UB[0], fields.UB[1],
            fields.UB[2], fields.U[0], fields.U[1], fields.U[2], fields.gradU, static_cast<long long>(fields.nCells), fields.nuEffB, fields.rhoB,
            fields.rhoP, fields.rhoV, fields.pRef, fields.correctGrad ? 1 : 0, fields.origin.x, fields.origin.y, fields.origin.z,
            faces.terms.data());
        cudaCheck(cudaGetLastError(), "forceFaceTerms");
        forceRunSumsKernel<<<sumBlocks, kSumThreads, 0, cudaStreamPerThread>>>(
            nPatches, faces.dPatchOffset.data(), c0, c1, c0 == 0 ? 1 : 0, reversePatch, faces.terms.data(), faces.runSums.data());
        cudaCheck(cudaGetLastError(), "forceRunSums");
    }
    forceRunTreeKernel<<<nPatches, kForceRuns, 0, cudaStreamPerThread>>>(faces.runSums.data(), faces.sums.data());
    cudaCheck(cudaGetLastError(), "forceRunTree");
    faces.sums.copyTo(out);
    return out;
}

} // namespace

bool forcesFaceParallelSelected()
{
    static const bool on = []()
    {
        const char* e = std::getenv("BRAE_FORCES_FACE_PARALLEL");
        const std::string v = e ? e : "";
        if (v.empty() || v == "0") return false;
        if (v == "1") return true;
        throw std::runtime_error("BRAE_FORCES_FACE_PARALLEL=" + v + ": 1 takes the forces' face-parallel integration, unset or 0 "
                                 "the one-block-a-patch kernel; no other value is read");
    }();
    return on;
}

std::vector<double> deviceForcePatchSums(
    const DeviceForceFaces& faces,
    const DeviceForceFields& fields)
{
    const int nPatches = static_cast<int>(faces.patchOffset.size()) - 1;
    std::vector<double> out(static_cast<std::size_t>(12 * (nPatches > 0 ? nPatches : 0)), 0.0);
    if (nPatches <= 0 || faces.nFaces == 0) return out;
    if (!fields.faceCellB || !fields.dcB || !fields.pB || !fields.UB[0] || !fields.U[0] || !fields.gradU || !fields.nuEffB || fields.nCells <= 0)
        throw std::logic_error("deviceForcePatchSums: a field the face integration reads is missing");
    bool faceParallel = fields.integration == ForceIntegration::faceParallel;
    if (fields.integration == ForceIntegration::environment && forcesFaceParallelSelected())
    {
        faceParallel = true;
        static bool announced = false;
        if (!announced)
        {
            announced = true;
            std::printf("  forces: face-parallel integration (BRAE_FORCES_FACE_PARALLEL=1): each face's terms in parallel, then "
                        "the same 256 runs a patch in face order and the same tree as the one-block-a-patch kernel, so the same "
                        "sums; unset it for that kernel\n");
        }
    }
    if (fields.controlReverseRun && !faceParallel)
        throw std::logic_error("deviceForcePatchSums: the reversed-run control is the face-parallel path's; the one-block-a-patch "
                               "kernel has no run to reverse");
    if (!faceParallel) return patchBlockSums(faces, fields, nPatches);
    return faceParallelSums(faces, fields, nPatches);
}

} // namespace brae
