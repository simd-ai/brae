// The reduction behind Foam::bound's message, and the message's own format.
//
// deviceMinMaxMeanInto is held to a host reference on sizes that straddle the block size and on the
// inputs bound() actually sees -- all-negative, one negative outlier, a single cell -- and for
// run-to-run determinism, because brae's reductions are fixed-order by design and a diagnostic whose
// numbers moved between identical runs would be worse than none.
//
// The FORMAT is held to real OpenFOAM output. The lines below are transcribed from a v2412
// rhoSimpleFoam log (squareBend, diagonal-preconditioned turbulence pair), not from bound.C's
// operator<< chain: the comma after the field name and the ABSENCE of one after each number are exactly
// what a from-memory transcription gets wrong, and being greppable beside OpenFOAM's own log is the
// whole value of the line.
#include "bound_report.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace brae;

static int fails = 0;

static void chk(bool ok, const char* what, double got, double want)
{
    std::printf("  %-54s %s  (got %.10g want %.10g)\n", what, ok ? "ok" : "FAIL", got, want);
    if (!ok) fails = 1;
}

// printBounding writes to stdout, so capture it to compare the bytes.
static std::string capture(const char* field, scalar mn, scalar mx, scalar av)
{
    std::fflush(stdout);
    std::FILE* f = std::tmpfile();
    if (!f) return std::string();
    const int saved = dup(fileno(stdout));
    dup2(fileno(f), fileno(stdout));
    printBounding(field, mn, mx, av);
    std::fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    std::rewind(f);
    char buf[256] = {0};
    const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    std::fclose(f);
    return std::string(buf);
}

int main()
{
    std::mt19937 rng(7);
    for (int n : {1, 2, 255, 256, 257, 1000, 306000})
    {
        std::vector<scalar> h(static_cast<std::size_t>(n));
        std::uniform_real_distribution<double> d(-1e3, 1e6);
        for (int i = 0; i < n; ++i) h[i] = d(rng);
        if (n > 3)
        {
            h[n/3] = -1.234e5;      // the negative outlier bound() exists for
            h[2*n/3] = 9.87e7;
        }
        scalar mn = h[0];
        scalar mx = h[0];
        long double sum = 0;
        for (int i = 0; i < n; ++i)
        {
            mn = std::fmin(mn, h[i]);
            mx = std::fmax(mx, h[i]);
            sum += h[i];
        }
        const scalar mean = static_cast<scalar>(sum / n);
        DeviceBuffer<scalar> x(h), out(3);
        deviceMinMaxMeanInto(x, out.data());
        std::vector<scalar> r;
        out.copyTo(r);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "n=%d min", n);
        chk(r[0] == mn, buf, r[0], mn);
        std::snprintf(buf, sizeof(buf), "n=%d max", n);
        chk(r[1] == mx, buf, r[1], mx);
        std::snprintf(buf, sizeof(buf), "n=%d mean", n);
        chk(std::fabs(r[2] - mean) <= 1e-12 * std::fabs(mean), buf, r[2], mean);
    }

    // All-negative: bound() fires on exactly this, and a min-reduction seeded with 0 would miss it.
    {
        const int n = 5000;
        std::vector<scalar> h(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) h[i] = -1.0 - i * 1e-3;
        DeviceBuffer<scalar> x(h), out(3);
        deviceMinMaxMeanInto(x, out.data());
        std::vector<scalar> r;
        out.copyTo(r);
        chk(r[0] == h[n-1], "all-negative min", r[0], h[n-1]);
        chk(r[1] == h[0], "all-negative max", r[1], h[0]);
    }

    // Determinism: the reduction is fixed-order, so two calls on one input agree bit for bit.
    {
        const int n = 306000;
        std::vector<scalar> h(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) h[i] = std::sin(0.37 * i) * 1e5;
        DeviceBuffer<scalar> x(h), a(3), b(3);
        deviceMinMaxMeanInto(x, a.data());
        deviceMinMaxMeanInto(x, b.data());
        std::vector<scalar> ra, rb;
        a.copyTo(ra);
        b.copyTo(rb);
        chk(ra[0] == rb[0] && ra[1] == rb[1] && ra[2] == rb[2],
            "two calls agree bit for bit", ra[2], rb[2]);
    }

    // The format, against lines taken from a real OpenFOAM log.
    {
        struct Case
        {
            const char* field;
            double      mn;
            double      mx;
            double      av;
            const char* want;
        };
        const Case cases[] = {
            {"epsilon", -33532.1,  1.2543e+07,  400769.0,    "bounding epsilon, min: -33532.1 max: 1.2543e+07 average: 400769\n"},
            {"k",       -2.93301,  21582.9,     1876.82,     "bounding k, min: -2.93301 max: 21582.9 average: 1876.82\n"},
            {"epsilon", -711148.0, 5.94098e+07, 2.10287e+06, "bounding epsilon, min: -711148 max: 5.94098e+07 average: 2.10287e+06\n"},
            {"k",       -45.5957,  20868.3,     2449.09,     "bounding k, min: -45.5957 max: 20868.3 average: 2449.09\n"},
        };
        for (const Case& c : cases)
        {
            const std::string got = capture(c.field, c.mn, c.mx, c.av);
            const bool ok = (got == std::string(c.want));
            std::printf("  %-54s %s\n",
                        "format is byte-identical to the OpenFOAM log line",
                        ok ? "ok" : "FAIL");
            if (!ok)
            {
                std::printf("      got  %s      want %s", got.c_str(), c.want);
                fails = 1;
            }
        }
    }

    std::printf(fails ? "== FAILED ==\n" : "== PASSED ==\n");
    return fails;
}
