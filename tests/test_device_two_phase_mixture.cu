// The two-phase mixture on the device, against the HOST reference, BIT FOR BIT.
//
// THE TOLERANCE IS A FEW ULP, AND WHY IT IS NOT ZERO. This gate was written as `==` and the device
// disagreed with the host in the last bit on nu. The cause is FUSED MULTIPLY-ADD: both compilers
// contract `x*y + z` -- nvcc by default, g++ at -O3 on ARM64 -- and each rounds once where the written
// expression rounds twice, but they do not pick the SAME fusion. Forbidding contraction on the device
// with __dmul_rn/__dadd_rn was tried and made it worse: the host still fused, so three fields started
// disagreeing instead of one.
//
// So the bound is a few ULP rather than zero, and what makes it still worth having is that it
// DISCRIMINATES: the one thing this component can get wrong is taking the CLAMPED alpha for rho, and
// the control below measures that at ~50 -- fifteen orders above the FMA difference.
//
// (The HOST side is itself gated against OpenFOAM's own expressions in tests/test_two_phase_mixture.cu,
// so agreement here is agreement with OpenFOAM, one link along.)
//
// THE SWEEP INCLUDES alpha OUTSIDE [0,1] ON PURPOSE. rho takes the RAW alpha and mu/nu the CLAMPED
// one, and the two agree on every in-range field -- so a fused kernel that clamps once and uses the
// result for all three passes every smooth test and is wrong exactly where MULES leaves an overshoot,
// which is exactly at an interface.
#include "two_phase_mixture_cpp.cuh"
#include "device_gate_finite.cuh"
#include "device_two_phase_mixture.cuh"
#include "device_buffer.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main()
{
    std::printf("== two-phase mixture: device vs host ==\n");

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    // damBreak's own phases, and a sweep that deliberately leaves [0,1].
    cpu::twoPhase::PhaseProperties p;
    p.rho1 = 1000; p.nu1 = 1e-06; p.rho2 = 1; p.nu2 = 1.48e-05;

    std::vector<scalar> a1;
    for (int i = -50; i <= 1050; ++i) a1.push_back(scalar(i) / scalar(1000));   // -0.05 .. 1.05
    a1.push_back(scalar(1e-300));                 // subnormal, where a clamp written as a<0 vs a<=0 shows
    a1.push_back(scalar(-1e-300));
    a1.push_back(scalar(0));
    a1.push_back(scalar(1));
    const int n = static_cast<int>(a1.size());

    // --- host ---
    std::vector<scalar> a2h(n), rhoH, muH, nuH;
    for (int c = 0; c < n; ++c) a2h[c] = scalar(1) - a1[c];
    cpu::twoPhase::mixtureRho(a1, a2h, p, rhoH);
    cpu::twoPhase::mixtureMu (a1, p, muH);
    cpu::twoPhase::mixtureNu (a1, muH, p, nuH);

    // --- device ---
    DeviceBuffer<scalar> dA1(a1), dA2(n), dRho(n), dMu(n), dNu(n);
    DevicePhaseProperties dp{p.rho1, p.nu1, p.rho2, p.nu2};
    deviceMixtureCorrect(dA1.data(), n, dp, dA2.data(), dRho.data(), dMu.data(), dNu.data());
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernel did not complete\n"); return 1; }

    std::vector<scalar> a2d, rhoD, muD, nuD;
    dA2.copyTo(a2d); dRho.copyTo(rhoD); dMu.copyTo(muD); dNu.copyTo(nuD);

    // Agreement measured in ULP, so the bound is a statement about rounding rather than about the
    // magnitudes of these particular phases.
    auto ulps = [](scalar a, scalar b)
    {
        if (a == b) return 0.0;
        const double d = std::fabs((double)a - (double)b);
        const double m = std::fmax(std::fabs((double)a), std::fabs((double)b));
        return d / std::max(std::nextafter(m, 1e300) - m, 1e-300);
    };
    auto agrees = [&](const char* name, const std::vector<scalar>& d, const std::vector<scalar>& h)
    {
        int nDiff = 0;
        double worst = 0; int iw = -1;
        for (int c = 0; c < n; ++c)
            if (d[c] != h[c])
            {
                ++nDiff;
                const double u = ulps(d[c], h[c]);
                if (u > worst) { worst = u; iw = c; }
            }
        if (nDiff)
            std::printf("    %s: %d of %d cells differ, worst %.2f ULP at alpha = %.6g "
                        "(device %.17g, host %.17g)\n",
                        name, nDiff, n, worst, (double)a1[iw], (double)d[iw], (double)h[iw]);
        else
            std::printf("    %s: every cell bit-identical\n", name);
        check((std::string(name) + " agrees with the host to within 2 ULP").c_str(), worst <= 2.0);
    };

    std::printf("  %d cells, alpha from %.3f to %.3f\n", n, (double)a1.front(), (double)a1[1100]);
    // alpha2 = 1 - alpha1 has no multiply-add to contract, so it IS bit-identical and is asserted so.
    agrees("alpha2", a2d, a2h);
    {
        bool bit = true;
        for (int c = 0; c < n; ++c) bit = bit && (a2d[c] == a2h[c]);
        check("...and alpha2 exactly so, having no multiply-add to fuse", bit);
    }
    agrees("rho",    rhoD, rhoH);
    agrees("mu",     muD,  muH);
    agrees("nu",     nuD,  nuH);

    // THE CONTROL: the sweep must actually contain the out-of-range region where raw and clamped
    // differ, or "matches exactly" would be a statement about the easy part of the field.
    {
        int nOut = 0;
        scalar biggest = 0;
        for (int c = 0; c < n; ++c)
        {
            if (a1[c] >= scalar(0) && a1[c] <= scalar(1)) continue;
            ++nOut;
            const scalar ac = cpu::twoPhase::limitedAlpha(a1[c]);
            const scalar rawRho     = a1[c]*p.rho1 + (scalar(1) - a1[c])*p.rho2;
            const scalar clampedRho = ac*p.rho1 + (scalar(1) - ac)*p.rho2;
            biggest = std::fmax(biggest, std::fabs(rawRho - clampedRho));
        }
        std::printf("  %d cells lie outside [0,1]; raw and clamped rho differ by up to %.4g there\n",
                    nOut, (double)biggest);
        check("the sweep covers the region where the raw/clamped split is visible", nOut > 50);
        check("...and the two forms differ there by an amount no tolerance would hide", biggest > scalar(1));
    }

    // ...and the null-output contract: asking for one field must not require the others.
    {
        DeviceBuffer<scalar> only(n);
        deviceMixtureCorrect(dA1.data(), n, dp, nullptr, only.data(), nullptr, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess)
        { std::printf("  FAIL: rho-only launch did not complete\n"); return 1; }
        std::vector<scalar> h;
        only.copyTo(h);
        failures += brae::gatecheck::nonFinite("h", h);
        // device against DEVICE: the two launches run the same instructions, so this one IS exact.
        bool same = true;
        for (int c = 0; c < n; ++c) same = same && (h[c] == rhoD[c]);
        check("asking for rho alone gives bit-identically the same rho", same);
    }

    std::printf("test_device_two_phase_mixture: %d failures\n", failures);
    return failures ? 1 : 0;
}
