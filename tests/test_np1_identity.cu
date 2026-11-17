// The single-GPU solver entry points on a synthetic box, printed as bit hashes: the SAME source is built in the tree before
// the distributed-solver hooks (reductions hook, processor-interface view, normFactor, V-cycle strip) and in the tree
// after, and the two outputs must be identical, line for line. With -DBRAE_DIST_HOOKS (this tree only) a ProcComm of ONE rank is
// installed first (ProcComm::setCurrent, a DeviceProcInterface on its zero-neighbour plan, view.proc assigned), which is
// what a rank of a np = 1 decomposed run does: that must be the same numbers too.
//
//   test_np1_identity [comm]       env: BRAE_PCG_DEVICE=0, BRAE_GS_HOST_SMOOTHER=0, ... select the solver paths
#include "device_amg.cuh"
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "proc_comm.cuh"
#ifdef BRAE_DIST_HOOKS
#include "device_proc_interface.cuh"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

namespace {

uint64_t fnv(const std::vector<scalar>& v)
{
    uint64_t h = 1469598103934665603ull;
    for (scalar x : v)
    {
        uint64_t b;
        std::memcpy(&b, &x, 8);
        for (int k = 0; k < 8; ++k) { h ^= (b >> (8*k)) & 0xff; h *= 1099511628211ull; }
    }
    return h;
}
uint64_t fnv1(scalar x) { return fnv(std::vector<scalar>{x}); }

struct Prob { int nx = 36, ny = 28, nz = 20; };

}

int main(int argc, char** argv)
{
    const bool withComm = argc > 1 && std::string(argv[1]) == "comm";
    Prob p;
    const int N = p.nx*p.ny*p.nz;
    std::vector<scalar> diag(N), b(N), x0(N), upper, lower;
    std::vector<label> owner, nei;
    auto w = [&](int g1, int g2) { const int d = g2 - g1; const double k = d == 1 ? 1.0 : d == p.nx ? 4.0 : 0.25; return k*(1.0 + 0.5*std::sin(0.37*g1 + 0.11*g2)); };
    for (int g = 0; g < N; ++g)
    {
        const int i = g % p.nx, j = (g / p.nx) % p.ny, k = g / (p.nx*p.ny);
        diag[g] = 0.2*(1.0 + 0.4*std::cos(0.021*g));
        b[g] = std::sin(0.013*g) + 0.1;
        x0[g] = std::cos(0.002*g) + 0.25;
        const int nb[6] = {i > 0 ? g - 1 : -1, i < p.nx - 1 ? g + 1 : -1, j > 0 ? g - p.nx : -1, j < p.ny - 1 ? g + p.nx : -1,
                           k > 0 ? g - p.nx*p.ny : -1, k < p.nz - 1 ? g + p.nx*p.ny : -1};
        for (int g2 : nb) if (g2 >= 0) diag[g] += w(std::min(g, g2), std::max(g, g2));
        for (int pass : {1, p.nx, p.nx*p.ny})                     // faces owned by g, ascending neighbour
        {
            const int g2 = g + pass;
            const bool ok = (pass == 1) ? i < p.nx - 1 : (pass == p.nx) ? j < p.ny - 1 : k < p.nz - 1;
            if (ok) { owner.push_back(g); nei.push_back(g2); upper.push_back(-w(g, g2)); lower.push_back(-w(g, g2)); }
        }
    }
    // faces of one owner ascending in neighbour: g+1 < g+nx < g+nx*ny already
    std::vector<scalar> faceW(upper.size());
    for (std::size_t f = 0; f < upper.size(); ++f) faceW[f] = -upper[f];

#ifdef BRAE_DIST_HOOKS
    std::vector<std::unique_ptr<ProcComm>> comms;
    std::unique_ptr<HaloPlan> plan;
    std::unique_ptr<DeviceProcInterface> ifc;
    if (withComm)
    {
        comms = ProcComm::createLoopback(1);
        ProcComm::setCurrent(comms[0].get());
        HaloPlanDesc desc;
        desc.nCells = N;
        plan = comms[0]->makePlan(desc);
        ifc = std::make_unique<DeviceProcInterface>(*comms[0], *plan);
    }
#else
    (void)withComm;
#endif
    DeviceLduMatrix M = buildDeviceLdu(diag, upper, lower, owner, nei, N);
    DeviceLduView A = M.view();
#ifdef BRAE_DIST_HOOKS
    if (ifc) A.proc = ifc->view();
#endif
    DeviceBuffer<scalar> db(b);
    AMGData amg = buildAMG(owner, nei, faceW, N);
    amgGalerkin(amg, M.diag, M.upper, M.lower);

    {
        DeviceBuffer<scalar> psi(x0), Ax, r;
        deviceAmul(A, psi, Ax, true);
        deviceResidual(A, psi, db, r, true);
        std::printf("amul      %016llx\nresidual  %016llx\n", (unsigned long long)fnv(Ax.host()), (unsigned long long)fnv(r.host()));
        const scalar nf = deviceNormFactor(A, psi, db, deviceOnes(N));
        std::printf("normFactor %016llx\n", (unsigned long long)fnv1(nf));
        DeviceBuffer<scalar> dd(2);
        deviceDotInto(psi, db, dd.data());
        deviceSumMagInto(psi, dd.data() + 1);
        std::printf("dotInto   %016llx\n", (unsigned long long)fnv(dd.host()));
        std::printf("dot       %016llx sumMag %016llx maxRatio %016llx\n", (unsigned long long)fnv1(deviceDot(psi, db)),
                    (unsigned long long)fnv1(deviceSumMag(psi)), (unsigned long long)fnv1(deviceMaxRatio(M.diag, deviceOnes(N))));
        DeviceBuffer<scalar> mm(3);
        deviceMinMaxMeanInto(psi, mm.data());
        std::printf("minmax    %016llx\n", (unsigned long long)fnv(mm.host()));
    }
    auto report = [&](const char* name, const DeviceSolverPerf& pf, DeviceBuffer<scalar>& x)
    {
        std::printf("%-10s iters %4d init %016llx final %016llx x %016llx\n", name, pf.nIterations,
                    (unsigned long long)fnv1(pf.initialResidual), (unsigned long long)fnv1(pf.finalResidual),
                    (unsigned long long)fnv(x.host()));
    };
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = deviceNormFactor(A, x, db, deviceOnes(N));
        const DeviceSolverPerf pf = deviceAMGPCG(A, amg, db, x, nf, 1e-10, 0.0, 500);
        report("amgpcg", pf, x);
    }
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = deviceNormFactor(A, x, db, deviceOnes(N));
        const DeviceSolverPerf pf = deviceAMGPCG(A, amg, db, x, nf, 1e-6, 0.1, 500, true, 1, false, 2);
        report("amgpcg-rel", pf, x);
    }
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = deviceNormFactor(A, x, db, deviceOnes(N));
        const DeviceSolverPerf pf = deviceJacobiPCG(A, db, x, nf, 1e-8, 0.0, 3000);
        report("jacobipcg", pf, x);
    }
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = deviceNormFactor(A, x, db, deviceOnes(N));
        const DeviceSolverPerf pf = deviceJacobiBiCGStab(A, db, x, nf, 1e-10, 0.0, 500, 1, 0, nullptr, &amg, 1);
        report("bicgstab", pf, x);
    }
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = deviceNormFactor(A, x, db, deviceOnes(N));
        DeviceSolverPerf pf;
        deviceSymGaussSeidel(A, db, x, nf, 1e-8, 0.0, 5000, &pf);
        report("symgs", pf, x);
    }
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = deviceNormFactor(A, x, db, deviceOnes(N));
        DeviceSolverPerf pf;
        deviceSymGaussSeidel(A, db, x, nf, 1e-3, 0.05, 50, &pf, 0, 2, true);
        report("symgs-rel", pf, x);
    }
#ifdef BRAE_DIST_HOOKS
    ifc.reset();
    plan.reset();
    ProcComm::setCurrent(nullptr);
#endif
    return 0;
}
