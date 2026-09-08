// deviceReadScalar's mailbox: the value it returns must be EXACTLY the value the blocking cudaMemcpy it
// replaced would have returned, for any producer. The oracle is that blocking copy, written out here so
// the test does not take the thing under test's word for it.
//
// Five parts, and the fourth is the control:
//   (a) bit-for-bit against the oracle over many values (including -0, denormals, infinities and NaNs with
//       payloads) and many repetitions;
//   (b) ordering: a SLOW producer enqueued on the per-thread stream ahead of each read, so a publish that
//       ran early would return the previous iteration's value;
//   (c) the two modes (mailbox and BRAE_READ_SCALAR_SYNC=1) agree bit for bit -- the second mode runs as a
//       child process, because the mode is latched once per process the way brae latches its other modes;
//   (d) CONTROL: the same slow producer on an INDEPENDENT stream, which nothing orders against the read.
//       Both modes must return the OLD value there. (b) fails if the read stops respecting stream order;
//       (d) fails if it starts synchronising more than the stream (a device-wide sync would pick the new
//       value up). Neither passes for the other's reason.
//   (e) the two-second bound is reachable and throws with its own name in the message.
//   (f) the MULTI-VALUE read (deviceReadValues): a group of scalars and ints published by one kernel and
//       taken in one wait, bit for bit against the same blocking oracle, over many repetitions -- at the
//       widths the solvers use (two scalars and an int; the six of the fused Gauss-Seidel report);
//   (g) ordering for the group: one SLOW producer writes every member of the group, so a publish that
//       ran early would return the previous iteration's values. (f) alone cannot see that.
// (f) and (g) push their values into the same `seen` vector as (a)/(b)/(d), so (c) holds the two modes
// to the same bits over them too.
#include "cf_types.cuh"
#include "device_blas.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace {

// The producer's spin, in SM clocks. 2e6 is about 1.3 ms on this box, which is two orders of magnitude
// more than a kernel launch, so a publish that did not wait for the producer would land inside the spin
// and read the previous value every time rather than sometimes.
constexpr long long SPIN_SHORT = 2000000LL;
constexpr long long SPIN_LONG = 400000000LL;   // ~0.25 s: the unordered control's producer must still be running

void check(
    cudaError_t e,
    const char* what)
{
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string("test_scalar_mailbox: ") + what + ": " + cudaGetErrorString(e));
    }
}

unsigned long long bitsOf(scalar v)
{
    unsigned long long b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

__global__
void writeBitsKernel(
    unsigned long long* d,
    unsigned long long bits)
{
    *d = bits;
}

// Writes only after spinning, so "did the read wait for this kernel" is a question with a visible answer.
__global__
void spinWriteBitsKernel(
    unsigned long long* d,
    unsigned long long bits,
    long long cycles)
{
    const long long t0 = clock64();
    while (clock64() - t0 < cycles)
    {
    }
    *d = bits;
}

// The oracle: the blocking D2H copy deviceReadScalar used to be, on the same per-thread stream, so it
// carries the same ordering guarantee and nothing more.
unsigned long long oracleRead(const unsigned long long* d)
{
    unsigned long long h = 0;
    check(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost), "oracle memcpy");
    return h;
}

// The group the solvers read: some 8-byte scalars and some 4-byte ints, laid out as the caller's own
// separate device scalars are. Writes each member from one thread so the whole group is one snapshot.
__global__
void writeGroupKernel(
    unsigned long long* ds,
    int nS,
    int* di,
    int nI,
    unsigned long long bits0,
    int i0,
    long long cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    const long long t0 = clock64();
    while (clock64() - t0 < cycles)
    {
    }
    for (int k = 0; k < nS; ++k)
    {
        ds[k] = bits0 + 0x1111111111111111ull * (unsigned long long)k;
    }
    for (int k = 0; k < nI; ++k)
    {
        di[k] = i0 + 7919 * k;
    }
}

// The blocking oracle for a group: the run of async D2H copies with one sync behind it that every
// converted site used to write out by hand.
void oracleReadGroup(
    const unsigned long long* ds,
    int nS,
    const int* di,
    int nI,
    unsigned long long* hs,
    int* hi)
{
    for (int k = 0; k < nS; ++k)
    {
        check(cudaMemcpyAsync(hs + k, ds + k, sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                              cudaStreamPerThread), "oracle group scalar");
    }
    for (int k = 0; k < nI; ++k)
    {
        check(cudaMemcpyAsync(hi + k, di + k, sizeof(int), cudaMemcpyDeviceToHost,
                              cudaStreamPerThread), "oracle group int");
    }
    check(cudaStreamSynchronize(cudaStreamPerThread), "oracle group sync");
}

// The values every mode reads, in order. Any bit pattern is a valid double; the comparison is on bits, so
// the NaNs and the two zeros are as meaningful as the finite numbers.
std::vector<unsigned long long> testBits()
{
    std::vector<scalar> v = {
        scalar(0.0),
        scalar(-0.0),
        scalar(1.0),
        scalar(-1.0),
        scalar(3.14159265358979311599796346854),
        scalar(1e-300),
        scalar(1e308),
        scalar(-1e308),
        scalar(2.2250738585072014e-308),
        scalar(1.0) / scalar(0.0),
        scalar(-1.0) / scalar(0.0)
    };
    std::vector<unsigned long long> b;
    for (scalar x : v)
    {
        b.push_back(bitsOf(x));
    }
    b.push_back(0x0000000000000001ull);   // smallest denormal
    b.push_back(0x7ff8000000012345ull);   // quiet NaN with a payload
    b.push_back(0x7ff0000000000001ull);   // signalling NaN
    b.push_back(0xfff8000000000001ull);   // negative NaN
    unsigned long long r = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < 48; ++i)
    {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        b.push_back(r);
    }
    return b;
}

// Every read this binary makes, in a fixed order, so a run in one mode can be compared line by line with a
// run in the other. `fails` counts the checks that a single mode can decide on its own.
std::vector<unsigned long long> runReads(int& fails)
{
    const std::vector<unsigned long long> vals = testBits();
    std::vector<unsigned long long> seen;
    unsigned long long* d = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(unsigned long long)), "alloc");

    // (a) plain producer, many repetitions, against the oracle.
    const int reps = 20;
    int aFails = 0;
    for (int rep = 0; rep < reps; ++rep)
    {
        for (unsigned long long want : vals)
        {
            writeBitsKernel<<<1, 1, 0, cudaStreamPerThread>>>(d, want);
            check(cudaGetLastError(), "writeBits");
            const scalar got = deviceReadScalar(reinterpret_cast<const scalar*>(d));
            const unsigned long long gotBits = bitsOf(got);
            const unsigned long long oracle = oracleRead(d);
            seen.push_back(gotBits);
            if (gotBits != want || gotBits != oracle)
            {
                if (aFails < 5)
                {
                    std::printf("  (a) MISMATCH rep %d: wanted %016llx, read %016llx, oracle %016llx\n",
                                rep, want, gotBits, oracle);
                }
                ++aFails;
            }
        }
    }
    std::printf("  (a) %d reads of %zu values vs the blocking oracle : %d mismatches\n",
                reps * static_cast<int>(vals.size()), vals.size(), aFails);
    fails += aFails;

    // (b) each read preceded on the SAME stream by a slow producer that overwrites what the previous read
    // saw. A publish that did not wait would return the previous iteration's value.
    int bFails = 0;
    const int nSeq = 24;
    for (int i = 0; i < nSeq; ++i)
    {
        const unsigned long long want = vals[static_cast<std::size_t>(i) % vals.size()] ^ (0x5bull * (i + 1));
        spinWriteBitsKernel<<<1, 1, 0, cudaStreamPerThread>>>(d, want, SPIN_SHORT);
        check(cudaGetLastError(), "spinWriteBits");
        const unsigned long long got = bitsOf(deviceReadScalar(reinterpret_cast<const scalar*>(d)));
        seen.push_back(got);
        if (got != want)
        {
            if (bFails < 5)
            {
                std::printf("  (b) ORDERING MISMATCH at %d: wanted %016llx, read %016llx\n", i, want, got);
            }
            ++bFails;
        }
    }
    std::printf("  (b) %d slow-producer reads, each overwriting the last : %d mismatches\n", nSeq, bFails);
    fails += bFails;

    // (f) the MULTI-VALUE read. Two widths: the two-scalars-and-a-count the graph solvers report, and the
    // six of the fused Gauss-Seidel report (GS_FUSED_MAX residuals and their sweep counts), which is the
    // widest read converted. Every member is checked against what the producer wrote AND against the
    // blocking oracle, so a helper that returned a plausible-looking wrong slot is caught.
    unsigned long long* dGroupS = nullptr;
    int* dGroupI = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&dGroupS), 3*sizeof(unsigned long long)), "group scalar alloc");
    check(cudaMalloc(reinterpret_cast<void**>(&dGroupI), 3*sizeof(int)), "group int alloc");
    const int widthS[2] = {2, 3};
    const int widthI[2] = {1, 3};
    int fFails = 0;
    int fReads = 0;
    for (int w = 0; w < 2; ++w)
    {
        const int nS = widthS[w];
        const int nI = widthI[w];
        for (int rep = 0; rep < reps; ++rep)
        {
            const unsigned long long bits0 = vals[static_cast<std::size_t>(rep) % vals.size()]
                                           ^ (0xa5a5ull * (unsigned long long)(rep + 1));
            const int i0 = 1 + rep*13 + w*1000;
            writeGroupKernel<<<1, 1, 0, cudaStreamPerThread>>>(dGroupS, nS, dGroupI, nI, bits0, i0, 0);
            check(cudaGetLastError(), "writeGroup");
            unsigned long long gotS[3] = {0, 0, 0};
            int gotI[3] = {0, 0, 0};
            DeviceReadValue v[6];
            for (int k = 0; k < nS; ++k)
            {
                v[k] = {dGroupS + k, gotS + k, false};
            }
            for (int k = 0; k < nI; ++k)
            {
                v[nS + k] = {dGroupI + k, gotI + k, true};
            }
            deviceReadValues(v, nS + nI);
            unsigned long long oraS[3] = {0, 0, 0};
            int oraI[3] = {0, 0, 0};
            oracleReadGroup(dGroupS, nS, dGroupI, nI, oraS, oraI);
            for (int k = 0; k < nS; ++k)
            {
                const unsigned long long want = bits0 + 0x1111111111111111ull * (unsigned long long)k;
                seen.push_back(gotS[k]);
                ++fReads;
                if (gotS[k] != want || gotS[k] != oraS[k])
                {
                    if (fFails < 5)
                    {
                        std::printf("  (f) MISMATCH width %d+%d rep %d scalar %d: wanted %016llx, read %016llx, oracle %016llx\n",
                                    nS, nI, rep, k, want, gotS[k], oraS[k]);
                    }
                    ++fFails;
                }
            }
            for (int k = 0; k < nI; ++k)
            {
                const int want = i0 + 7919*k;
                seen.push_back(static_cast<unsigned long long>(static_cast<unsigned int>(gotI[k])));
                ++fReads;
                if (gotI[k] != want || gotI[k] != oraI[k])
                {
                    if (fFails < 5)
                    {
                        std::printf("  (f) MISMATCH width %d+%d rep %d int %d: wanted %d, read %d, oracle %d\n",
                                    nS, nI, rep, k, want, gotI[k], oraI[k]);
                    }
                    ++fFails;
                }
            }
        }
    }
    std::printf("  (f) %d grouped values (2+1 and 3+3 per wait) vs the blocking oracle : %d mismatches\n",
                fReads, fFails);
    fails += fFails;

    // (g) ORDERING for the group: one SLOW producer writes every member, and each pass overwrites what the
    // last read saw. A publish that ran before the producer would return the previous pass's values -- the
    // failure mode (f) cannot see, because (f)'s producer finishes long before anything asks for it.
    int gFails = 0;
    const int nSeqG = 16;
    for (int i = 0; i < nSeqG; ++i)
    {
        const unsigned long long bits0 = vals[static_cast<std::size_t>(i) % vals.size()]
                                       ^ (0x3c3cull * (unsigned long long)(i + 1));
        const int i0 = 5000 + 31*i;
        writeGroupKernel<<<1, 1, 0, cudaStreamPerThread>>>(dGroupS, 3, dGroupI, 3, bits0, i0, SPIN_SHORT);
        check(cudaGetLastError(), "spinWriteGroup");
        unsigned long long gotS[3] = {0, 0, 0};
        int gotI[3] = {0, 0, 0};
        DeviceReadValue v[6];
        for (int k = 0; k < 3; ++k)
        {
            v[k] = {dGroupS + k, gotS + k, false};
            v[3 + k] = {dGroupI + k, gotI + k, true};
        }
        deviceReadValues(v, 6);
        for (int k = 0; k < 3; ++k)
        {
            const unsigned long long wantS = bits0 + 0x1111111111111111ull * (unsigned long long)k;
            const int wantI = i0 + 7919*k;
            seen.push_back(gotS[k]);
            seen.push_back(static_cast<unsigned long long>(static_cast<unsigned int>(gotI[k])));
            if (gotS[k] != wantS || gotI[k] != wantI)
            {
                if (gFails < 5)
                {
                    std::printf("  (g) ORDERING MISMATCH at %d slot %d: wanted %016llx/%d, read %016llx/%d\n",
                                i, k, wantS, wantI, gotS[k], gotI[k]);
                }
                ++gFails;
            }
        }
    }
    std::printf("  (g) %d slow-producer grouped reads of 6 values, each overwriting the last : %d mismatches\n",
                nSeqG, gFails);
    fails += gFails;
    check(cudaFree(dGroupS), "group scalar free");
    check(cudaFree(dGroupI), "group int free");

    // (d) CONTROL. The producer goes on an independent stream, so nothing orders it against the read. The
    // blocking copy does not wait for another stream (measured on this box), and neither may the mailbox:
    // both must still see the OLD value. A read that synchronised the device instead of the stream would
    // see the new one and fail here while still passing (b).
    const unsigned long long before = 0x0123456789abcdefull;
    const unsigned long long after = 0xfedcba9876543210ull;
    writeBitsKernel<<<1, 1, 0, cudaStreamPerThread>>>(d, before);
    check(cudaStreamSynchronize(cudaStreamPerThread), "control seed sync");
    cudaStream_t other;
    check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "control stream");
    spinWriteBitsKernel<<<1, 1, 0, other>>>(d, after, SPIN_LONG);
    check(cudaGetLastError(), "control producer");
    const unsigned long long unordered = bitsOf(deviceReadScalar(reinterpret_cast<const scalar*>(d)));
    seen.push_back(unordered);
    check(cudaDeviceSynchronize(), "control drain");
    const unsigned long long settled = oracleRead(d);
    check(cudaStreamDestroy(other), "control stream destroy");
    const bool controlOk = unordered == before && settled == after;
    std::printf("  (d) control, unordered producer : read %016llx (old %016llx), settled %016llx (new %016llx) %s\n",
                unordered, before, settled, after, controlOk ? "" : "  <-- FAIL");
    if (!controlOk)
    {
        ++fails;
    }

    check(cudaFree(d), "free");
    return seen;
}

// (c) the other mode, as a child process: the mode is latched on the first read, so it cannot be switched
// inside one process. The child prints one "bits <hex>" line per read and nothing else is parsed.
bool childBits(
    const char* self,
    std::vector<unsigned long long>& out)
{
    std::string path = self;
    if (path.find('/') == std::string::npos)
    {
        path = "./" + path;
    }
    const std::string cmd = "BRAE_READ_SCALAR_SYNC=1 '" + path + "' --emit";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p)
    {
        return false;
    }
    char line[256];
    while (std::fgets(line, sizeof(line), p))
    {
        unsigned long long b = 0;
        if (std::sscanf(line, "bits %llx", &b) == 1)
        {
            out.push_back(b);
        }
    }
    return pclose(p) == 0;
}

} // namespace


int main(int argc, char** argv)
{
    const bool emit = argc > 1 && std::string(argv[1]) == "--emit";
    int fails = 0;
    const std::vector<unsigned long long> mine = runReads(fails);

    if (emit)
    {
        for (unsigned long long b : mine)
        {
            std::printf("bits %016llx\n", b);
        }
        return fails == 0 ? 0 : 1;
    }

    // (c) mode agreement.
    std::vector<unsigned long long> other;
    const bool childOk = childBits(argv[0], other);
    int cFails = 0;
    if (!childOk || other.size() != mine.size())
    {
        std::printf("  (c) BRAE_READ_SCALAR_SYNC=1 child: ok=%d, %zu reads vs %zu  <-- FAIL\n",
                    static_cast<int>(childOk), other.size(), mine.size());
        ++cFails;
    }
    else
    {
        for (std::size_t i = 0; i < mine.size(); ++i)
        {
            if (mine[i] != other[i])
            {
                if (cFails < 5)
                {
                    std::printf("  (c) MODE MISMATCH at %zu: mailbox %016llx, sync %016llx\n",
                                i, mine[i], other[i]);
                }
                ++cFails;
            }
        }
        std::printf("  (c) mailbox vs BRAE_READ_SCALAR_SYNC=1 over %zu reads : %d mismatches\n",
                    mine.size(), cFails);
    }
    fails += cFails;

    // (e) the bound. The probe waits on a sequence number no kernel will publish, so the wait has to give
    // up on the clock and say which wait it was.
    bool threw = false;
    bool named = false;
    const auto t0 = std::chrono::steady_clock::now();
    try
    {
        deviceReadScalarWaitProbe();
    }
    catch (const std::exception& e)
    {
        threw = true;
        named = std::string(e.what()).find("deviceReadScalar") != std::string::npos;
        std::printf("  (e) timeout threw: %s\n", e.what());
    }
    const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool eOk = threw && named && waited > 1.5 && waited < 10.0;
    std::printf("  (e) bounded wait : threw=%d named=%d after %.2f s %s\n",
                static_cast<int>(threw), static_cast<int>(named), waited, eOk ? "" : "  <-- FAIL");
    if (!eOk)
    {
        ++fails;
    }

    // A read after the timeout still works: the probe consumed no sequence number.
    unsigned long long* d = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(unsigned long long)), "post alloc");
    writeBitsKernel<<<1, 1, 0, cudaStreamPerThread>>>(d, 0xdeadbeefcafef00dull);
    const unsigned long long post = bitsOf(deviceReadScalar(reinterpret_cast<const scalar*>(d)));
    check(cudaFree(d), "post free");
    if (post != 0xdeadbeefcafef00dull)
    {
        std::printf("  (e) a read after the timeout returned %016llx  <-- FAIL\n", post);
        ++fails;
    }

    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
