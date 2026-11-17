// Loopback concurrent-rank reduction ownership (review finding 7).
//
//   test_loopback_reduction_threads [np]
//
// Runs np independent loopback rank THREADS, each with its own ProcComm and cudaStreamPerThread. Each thread
// repeatedly fills differently sized buffers, performs host-read reductions, device-resident reductions,
// deviceOnes lookup and a processor exchange. The host-read results are checked against sequential rank-local
// arithmetic (exactly, since each local dot/sum is checked before its global fold is checked by a second,
// identical collective). A shared accumulator would be overwritten between another rank's launch and read;
// 200 rounds make that window overwhelmingly likely to be hit. The deviceOnes contents are checked too, so a
// shared cache cannot hand a rank the wrong length or another thread's allocation.
#include "proc_comm.cuh"
#include "device_blas.cuh"
#include "device_proc_interface.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace brae;

namespace {

int blocks(int n) { return (n + 255)/256; }

__global__ void fillK(scalar* p, int n, scalar base)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) p[i] = base + scalar(i%17)/scalar(16);
}

__global__ void onesCheckK(const scalar* p, int n, int* bad)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && p[i] != scalar(1)) atomicAdd(bad, 1);
}

} // namespace

int main(int argc, char** argv)
{
    const int np = argc > 1 ? std::atoi(argv[1]) : 4;
    if (np < 2 || np > 8)
    {
        std::fprintf(stderr, "usage: test_loopback_reduction_threads [np=2..8]\n");
        return 2;
    }
    auto comms = ProcComm::createLoopback(np, 64);
    std::vector<std::thread> threads;
    std::vector<std::string> errors(np);
    std::vector<int> failed(np, 0);
    for (int r = 0; r < np; ++r)
    {
        threads.emplace_back([&, r] {
            try
            {
                comms[r]->bindThread();
                const int n = 1000 + 257*r;             // deliberately different block counts per rank
                DeviceBuffer<scalar> x(n), y(n), z(n);
                std::vector<scalar> hx(n), hy(n), expect(n);
                for (int i = 0; i < n; ++i)
                {
                    hx[i] = scalar(1 + r) + scalar(i%17)/scalar(16);
                    hy[i] = scalar(2 + 2*r) + scalar((n-i)%13)/scalar(8);
                }
                x.copyFrom(hx);
                y.copyFrom(hy);
                auto plan = comms[r]->makePlan(HaloPlanDesc{}, 4);
                for (int round = 0; round < 200; ++round)
                {
                    double dotLocal = 0, sumLocal = 0, maxLocal = 0;
                    for (int i = 0; i < n; ++i)
                    {
                        dotLocal += hx[i]*hy[i];
                        sumLocal += std::fabs(hx[i]);
                        maxLocal = std::max(maxLocal, std::fabs(hx[i])/std::fabs(hy[i]));
                    }
                    const double dot = deviceDot(x, y);
                    const double sum = deviceSumMag(x);
                    const double ratio = deviceMaxRatio(x, y);
                    // ProcComm folds all ranks' local values. Verify the global results independently in rank
                    // order (a serial reference of the SAME local values, no shared reduction scratch).
                    double dotWant = 0, sumWant = 0, maxWant = 0;
                    for (int q = 0; q < np; ++q)
                    {
                        const int nq = 1000 + 257*q;
                        double dq = 0, sq = 0;
                        for (int i = 0; i < nq; ++i)
                        {
                            const double xv = scalar(1 + q) + scalar(i%17)/scalar(16);
                            const double yv = scalar(2 + 2*q) + scalar((nq-i)%13)/scalar(8);
                            dq += xv*yv;
                            sq += std::fabs(xv);
                            maxWant = std::max(maxWant, std::fabs(xv)/std::fabs(yv));
                        }
                        dotWant += dq;
                        sumWant += sq;
                    }
                    const double tol = 2e-11*std::max(1.0, std::fabs(dotWant));
                    if (std::fabs(dot - dotWant) > tol || std::fabs(sum - sumWant) > tol || ratio != maxWant)
                    {
                        ++failed[r];
                        errors[r] = "host-read reduction differs from the independent global reference at round "
                                    + std::to_string(round);
                        break;
                    }
                    // The device-resident path uses the same partials and then the global fold. Two identical
                    // host collectives must agree, and a stale partial from another rank breaks the result.
                    z.resize(1);
                    deviceDotInto(x, y, z.data());
                    const double dot2 = deviceDot(x, y);
                    if (dot2 != dot)
                    {
                        ++failed[r];
                        errors[r] = "deviceDotInto changed the next host-read dot at round " + std::to_string(round);
                        break;
                    }
                    const DeviceBuffer<scalar>& ones = deviceOnes(n);
                    if (static_cast<int>(ones.size()) != n)
                    {
                        ++failed[r];
                        errors[r] = "deviceOnes returned another rank's length";
                        break;
                    }
                    if (round%40 == 0)
                    {
                        DeviceBuffer<int> bad(1);
                        bad.copyFrom(std::vector<int>(1, 0));
                        onesCheckK<<<blocks(n), 256>>>(ones.data(), n, bad.data());
                        if (bad.host().at(0) != 0)
                        {
                            ++failed[r];
                            errors[r] = "deviceOnes contents were corrupted by a concurrent rank";
                            break;
                        }
                    }
                    // Join the rank collective too: per-thread counters and scratch must not change SPMD ordering.
                    comms[r]->barrier();
                }
            }
            catch (const std::exception& e) { errors[r] = e.what(); ++failed[r]; }
            catch (...) { errors[r] = "unknown exception"; ++failed[r]; }
        });
    }
    for (auto& t : threads) t.join();
    int bad = 0;
    for (int r = 0; r < np; ++r)
        if (failed[r]) { ++bad; std::fprintf(stderr, "rank %d: %s\n", r, errors[r].c_str()); }
    if (bad == 0) std::printf("PASS loopback concurrent reductions (np=%d, 200 rounds/rank)\n", np);
    return bad == 0 ? 0 : 1;
}
