// See device_glibc_math.cuh. Compiled with -fmad=false (CMakeLists.txt): the ported code spells every operation out,
// and nothing else in this file may be contracted either.
#include "device_glibc_math.cuh"
#include "glibc_acos_cos.cuh"
#include "correctly_rounded_acos.cuh"
#include "device_buffer.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

// 0: glibc's acos and cos, FMA variants (glibc >= 2.34 on an FMA CPU)
// 1: ...the plain variants (glibc >= 2.34 with FMA off, or a CPU without it)
// 2: a correctly rounded acos and glibc's FMA cos (glibc <= 2.33 on an FMA CPU)
// 3: ...and with the plain cos
constexpr int nVariants = 4;

template <int V>
__device__ __forceinline__ double b2Of(double a12, double theta)
{
    if constexpr (V == 0) return glibcport::cos_glibc<true>(__dsub_rn(glibcport::acos_glibc<true>(a12), theta));
    if constexpr (V == 1) return glibcport::cos_glibc<false>(__dsub_rn(glibcport::acos_glibc<false>(a12), theta));
    if constexpr (V == 2) return glibcport::cos_glibc<true>(__dsub_rn(crdd::acos_cr(a12), theta));
    return glibcport::cos_glibc<false>(__dsub_rn(crdd::acos_cr(a12), theta));
}

template <int V>
__global__ void b2Kernel(int n, const label* __restrict__ kind, const scalar* __restrict__ theta,
                         const scalar* __restrict__ a12, scalar* __restrict__ b2)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    b2[i] = kind[i] == 1 ? b2Of<V>(a12[i], theta[i]) : scalar(0);
}

// the self-test's kernel: one angle for every sample
template <int V>
__global__ void testKernel(int n, double theta, const scalar* __restrict__ a12, scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = b2Of<V>(a12[i], theta);
}

void launchTest(int v, int n, double theta, const scalar* a12, scalar* out)
{
    const int blocks = (n + 255)/256;
    switch (v)
    {
        case 0: testKernel<0><<<blocks, 256>>>(n, theta, a12, out); break;
        case 1: testKernel<1><<<blocks, 256>>>(n, theta, a12, out); break;
        case 2: testKernel<2><<<blocks, 256>>>(n, theta, a12, out); break;
        default: testKernel<3><<<blocks, 256>>>(n, theta, a12, out); break;
    }
}

// deterministic samples over a12's real range [-1, 1]: uniform, small, and near +-1, where glibc's branches are
inline std::uint64_t mix64(std::uint64_t z)
{
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27))*0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::vector<scalar> testSamples(int n)
{
    std::vector<scalar> a(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const std::uint64_t h1 = mix64(2ULL*static_cast<std::uint64_t>(i));
        const std::uint64_t h2 = mix64(2ULL*static_cast<std::uint64_t>(i) + 1);
        const double u = static_cast<double>(h1 >> 11)*0x1p-53;
        const double sgn = ((h2 >> 62) & 1) ? -1.0 : 1.0;
        double v;
        switch (i & 15)
        {
            case 10: case 11: v = sgn*std::ldexp(1.0 + u, -static_cast<int>(h2 % 62) - 1); break;
            case 12: v = sgn*(1.0 - u*1e-15); break;
            case 13: v = sgn*(1.0 - std::ldexp(u, -static_cast<int>(h2 % 52) - 1)); break;
            case 14: v = sgn*(0.9 + 0.1*u); break;
            default: v = 2.0*u - 1.0; break;
        }
        a[static_cast<std::size_t>(i)] = v;
    }
    // exact values the branches turn on
    const double special[] = {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.9375, -0.9375, 0.96875, -0.96875,
                              std::nextafter(1.0, 0.0), -std::nextafter(1.0, 0.0), 1e-300, -1e-300, 1e-8, -1e-8};
    for (std::size_t k = 0; k < sizeof(special)/sizeof(special[0]) && k < a.size(); ++k) a[k] = special[k];
    return a;
}

// bit-for-bit mismatches of variant v against the host's libm, over the samples, at one angle
long testVariant(int v, int n, const std::vector<scalar>& a, const scalar* dA, scalar* dOut,
                 std::vector<scalar>& ref, std::vector<scalar>& got, double theta)
{
    launchTest(v, n, theta, dA, dOut);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (cudaMemcpy(got.data(), dOut, static_cast<std::size_t>(n)*sizeof(scalar), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    const long nl = n;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < nl; ++i)
    {
        ref[static_cast<std::size_t>(i)] = std::cos(std::acos(a[static_cast<std::size_t>(i)]) - theta);
    }
    long bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        if (std::memcmp(&ref[i], &got[i], sizeof(scalar)) != 0) ++bad;
    }
    return bad;
}

struct Choice
{
    bool done = false;
    int variant = -1;
    std::set<double> tested;
};
Choice g_choice;

long testAngles(int v, int n, const std::vector<double>& thetas)
{
    const std::vector<scalar> a = testSamples(n);
    DeviceBuffer<scalar> dA(a);
    DeviceBuffer<scalar> dOut(static_cast<std::size_t>(n));
    std::vector<scalar> ref(a.size()), got(a.size());
    long bad = 0;
    for (const double th : thetas)
    {
        const long b = testVariant(v, n, a, dA.data(), dOut.data(), ref, got, th);
        if (b < 0) return -1;
        bad += b;
    }
    return bad;
}

} // namespace

const char* glibcB2VariantName(int variant)
{
    switch (variant)
    {
        case 0: return "glibc's acos and cos, FMA variants";
        case 1: return "glibc's acos and cos, plain variants";
        case 2: return "a correctly rounded acos and glibc's FMA cos (glibc <= 2.33)";
        case 3: return "a correctly rounded acos and glibc's plain cos (glibc <= 2.33)";
        default: return "none: the host's libm";
    }
}

int glibcB2Variant(const std::vector<scalar>& thetas)
{
    static const double pi = 3.14159265358979323846;
    std::vector<double> fixed = {0.0, 0.3, pi/6.0, 1.2, pi/2.0, 2.0*pi/3.0, pi};
    constexpr int nSamples = 1 << 21;
    if (!g_choice.done)
    {
        g_choice.done = true;
        std::vector<double> all = fixed;
        for (const scalar t : thetas)
        {
            if (std::find(all.begin(), all.end(), static_cast<double>(t)) == all.end() && all.size() < 24) all.push_back(t);
        }
        for (int v = 0; v < nVariants; ++v)
        {
            if (testAngles(v, nSamples, all) == 0)
            {
                g_choice.variant = v;
                break;
            }
        }
        g_choice.tested.insert(all.begin(), all.end());
        return g_choice.variant;
    }
    if (g_choice.variant < 0) return -1;
    std::vector<double> fresh;
    for (const scalar t : thetas)
    {
        if (g_choice.tested.insert(static_cast<double>(t)).second && fresh.size() < 16) fresh.push_back(t);
    }
    if (!fresh.empty() && testAngles(g_choice.variant, 1 << 18, fresh) != 0)
    {
        g_choice.variant = -1;
    }
    return g_choice.variant;
}

void glibcB2(int variant, int n, const label* kind, const scalar* theta, const scalar* a12, scalar* b2)
{
    if (n <= 0) return;
    const int blocks = (n + 255)/256;
    switch (variant)
    {
        case 0: b2Kernel<0><<<blocks, 256>>>(n, kind, theta, a12, b2); break;
        case 1: b2Kernel<1><<<blocks, 256>>>(n, kind, theta, a12, b2); break;
        case 2: b2Kernel<2><<<blocks, 256>>>(n, kind, theta, a12, b2); break;
        case 3: b2Kernel<3><<<blocks, 256>>>(n, kind, theta, a12, b2); break;
        default: throw std::runtime_error("brae: glibcB2 called with no device variant");
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("brae: glibcB2 kernel: ") + cudaGetErrorString(e));
}

} // namespace brae
