// Distributed linear solvers on a decomposed mesh (multi-GPU interFoam, CommSolvers.md steps E-I), against the
// monolithic single-GPU solve of the SAME problem. No OpenFOAM: the problems are synthetic boxes of cells split by
// ProcComm's block partitioner (block and ragged "scattered" decompositions), assembled per rank as OpenFOAM's decomposePar
// output would give them -- a local lduMatrix of the rank's own cells plus one coefficient per processor face.
//
// One process per rank under mpirun (MPI-host backend), np = 1, 2, 4, 8 --oversubscribe on one GPU:
//     mpirun --oversubscribe -np N test_dist_solvers [bench]
// Rank THREADS (the loopback backend) are not used: the solvers keep process-wide caches (graph caches, level
// schedules, the reduction scratch, the single-threaded DevicePool), and MPI processes share none of them.
//
// What it holds:
//   (a) the hooks: the distributed product, residual and normFactor equal the monolithic ones; deviceDot / deviceSumMag /
//       deviceMaxRatio / deviceMinMaxMeanInto are GLOBAL and bit-identical on every rank; a LocalReductions scope is local
//   (b) AMG-PCG: the distributed solution equals the monolithic one to the solver tolerance; EVERY RANK reports the same
//       iteration count and the same residual history (bit for bit); the fused two-collective loop is bit-identical to the
//       plain three-collective one (BRAE_PROC_PCG_UNFUSED)
//   (c) symGaussSeidel with the halo frozen per sweep reaches the same solution; all ranks take the same sweep count
//   (d) the iteration count against np (printed), and the collectives / halo exchanges per PCG iteration (counted)
#include "device_amg.cuh"
#include "device_amg_detail.cuh"  // braeCondGraphsAvailable()
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_proc_interface.cuh"
#include "proc_hooks.cuh"
#include "cf_pstream.cuh"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace brae;

namespace {

int g_fail = 0;
ProcComm* g_comm = nullptr;

void check(bool ok, const std::string& what)
{
    if (!ok)
    {
        ++g_fail;
        std::printf("FAIL rank %d: %s\n", g_comm->rank(), what.c_str());
        std::fflush(stdout);
    }
}

// the same on every rank? (bit for bit)
bool sameOnAllRanks(const std::vector<double>& v)
{
    std::vector<double> lo = v, hi = v;
    g_comm->allReduceHost(lo.data(), static_cast<int>(lo.size()), ReduceOp::Min);
    g_comm->allReduceHost(hi.data(), static_cast<int>(hi.size()), ReduceOp::Max);
    return std::memcmp(lo.data(), hi.data(), lo.size()*sizeof(double)) == 0 && std::memcmp(lo.data(), v.data(), lo.size()*sizeof(double)) == 0;
}
bool sameOnAllRanks(double v) { return sameOnAllRanks(std::vector<double>{v}); }

// ---- the synthetic problem ------------------------------------------------------------------------------------
struct Problem
{
    const char* name;
    int nx, ny, nz;
    double kx, ky, kz;       // directional conductivities
    double mass;             // diagonal shift
};

inline double faceW(const Problem& p, int g1, int g2)
{
    const int d = g2 - g1;
    const double k = (d == 1) ? p.kx : (d == p.nx) ? p.ky : p.kz;
    return k*(1.0 + 0.5*std::sin(0.37*g1 + 0.11*g2));
}
inline double massAt(const Problem& p, int g) { return p.mass*(1.0 + 0.4*std::cos(0.021*g)); }
inline double rhsAt(int g) { return std::sin(0.013*g) + 0.3*std::cos(0.0071*g*g*1e-3) + 0.1; }
inline double fieldAt(int g) { return std::cos(0.002*g) + 0.25; }

// neighbours of global cell g inside the box
template <class F> void forNeighbours(const Problem& p, int g, F f)
{
    const int i = g % p.nx, j = (g / p.nx) % p.ny, k = g / (p.nx*p.ny);
    if (i > 0)        f(g - 1);
    if (i < p.nx - 1) f(g + 1);
    if (j > 0)        f(g - p.nx);
    if (j < p.ny - 1) f(g + p.nx);
    if (k > 0)        f(g - p.nx*p.ny);
    if (k < p.nz - 1) f(g + p.nx*p.ny);
}

struct LocalSystem
{
    int nC = 0, nHalo = 0;
    std::vector<int> cellGlobal;
    std::vector<scalar> diag, upper, lower, faceWeight, b, x0;
    std::vector<label> owner, nei;
    std::vector<scalar> procCoeff;       // one per processor face, the plan's order: -w
};

LocalSystem assemble(const Problem& p, const SyntheticPartition& P, int rank)
{
    LocalSystem s;
    s.cellGlobal = P.cellGlobal[rank];
    s.nC = static_cast<int>(s.cellGlobal.size());
    s.nHalo = static_cast<int>(P.haloGlobal[rank].size());
    const int N = p.nx*p.ny*p.nz;
    std::vector<int> g2l(N, -1);
    for (int l = 0; l < s.nC; ++l) g2l[s.cellGlobal[l]] = l;
    s.diag.assign(s.nC, 0.0);
    s.b.resize(s.nC);
    s.x0.resize(s.nC);
    struct Face { int l1, l2; double w; };
    std::vector<Face> faces;
    for (int l = 0; l < s.nC; ++l)
    {
        const int g = s.cellGlobal[l];
        s.diag[l] = massAt(p, g);
        s.b[l] = rhsAt(g);
        s.x0[l] = fieldAt(g);
        forNeighbours(p, g, [&](int g2)
        {
            const double w = faceW(p, std::min(g, g2), std::max(g, g2));
            s.diag[l] += w;                                  // every face of the box, interface faces included
            if (g2 > g && g2l[g2] >= 0) faces.push_back({l, g2l[g2], w});
        });
    }
    std::sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) { return a.l1 != b.l1 ? a.l1 < b.l1 : a.l2 < b.l2; });
    for (const Face& f : faces)
    {
        s.owner.push_back(f.l1);
        s.nei.push_back(f.l2);
        s.upper.push_back(-f.w);
        s.lower.push_back(-f.w);
        s.faceWeight.push_back(f.w);
    }
    const HaloPlanDesc& d = P.descs[rank];
    for (int i = 0; i < d.nProcFaces; ++i)
    {
        const int g = s.cellGlobal[d.faceCells[i]];
        const int g2 = P.haloGlobal[rank][d.faceToSlot[i]];
        s.procCoeff.push_back(-faceW(p, std::min(g, g2), std::max(g, g2)));
    }
    return s;
}

struct DeviceSystem
{
    DeviceLduMatrix M;
    DeviceBuffer<scalar> b;
    DeviceBuffer<scalar> x0;            // the smooth field, with the halo tail
    std::unique_ptr<DeviceProcInterface> ifc;
    DeviceLduView view;
    AMGData amg;

    DeviceSystem(const LocalSystem& s, ProcComm* comm, const HaloPlan* plan, const Problem& p)
    : b(s.b), x0(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo))
    {
        M = buildDeviceLdu(s.diag, s.upper, s.lower, s.owner, s.nei, s.nC);
        x0.copyFrom(s.x0);
        view = M.view();
        if (comm)
        {
            ifc = std::make_unique<DeviceProcInterface>(*comm, *plan);
            ifc->setCoeff(s.procCoeff);
            view.proc = ifc->view();
        }
        amg = buildAMG(s.owner, s.nei, s.faceWeight, s.nC);
        amgGalerkin(amg, M.diag, M.upper, M.lower);
        (void)p;
    }
};

std::vector<double> gatherToGlobal(const LocalSystem& s, const std::vector<scalar>& xl, int N)
{
    // every rank's block into the global vector (zeros elsewhere), summed over ranks
    std::vector<double> g(N, 0.0);
    for (int l = 0; l < s.nC; ++l) g[s.cellGlobal[l]] = xl[l];
    g_comm->allReduceHost(g.data(), N, ReduceOp::Sum);
    return g;
}

struct Solve
{
    DeviceSolverPerf perf;
    std::vector<scalar> x;
    std::vector<scalar> hist;
};

scalar normFactorOf(DeviceSystem& S, const DeviceBuffer<scalar>& psi, int nC)
{
    return deviceNormFactor(S.view, psi, S.b, deviceOnes(nC));
}

// graph = true: the whole-loop CUDA graph (NCCL backend only: BRAE_PROC_GRAPH=1); false: the host-driven loop, the
// oracle every other section of this test was written against
Solve pcgSolve(DeviceSystem& S, const LocalSystem& s, scalar tol, scalar relTol, int maxIter, bool unfused,
               int minIter = 0, bool graph = false)
{
    DeviceBuffer<scalar> x(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
    x.copyFrom(std::vector<scalar>(s.nC, 0.0));
    const scalar nf = normFactorOf(S, x, s.nC);
    if (unfused) setenv("BRAE_PROC_PCG_UNFUSED", "1", 1); else unsetenv("BRAE_PROC_PCG_UNFUSED");
    setenv("BRAE_PROC_GRAPH", graph ? "1" : "0", 1);
    Solve r;
    amgPcgSetHistory(&r.hist);
    r.perf = deviceAMGPCG(S.view, S.amg, S.b, x, nf, tol, relTol, maxIter, false, 1, false, minIter);
    amgPcgSetHistory(nullptr);
    unsetenv("BRAE_PROC_PCG_UNFUSED");
    r.x = x.host();
    return r;
}

Solve gsSolve(DeviceSystem& S, const LocalSystem& s, scalar tol, int maxIter, scalar relTol = 0.0, int minIter = 0,
              int nSweeps = 1, bool graph = false)
{
    DeviceBuffer<scalar> x(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
    x.copyFrom(std::vector<scalar>(s.nC, 0.0));
    const scalar nf = normFactorOf(S, x, s.nC);
    setenv("BRAE_PROC_GRAPH", graph ? "1" : "0", 1);
    Solve r;
    deviceSymGaussSeidel(S.view, S.b, x, nf, tol, relTol, maxIter, &r.perf, minIter, nSweeps);
    r.x = x.host();
    return r;
}

double maxRelDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double d = 0, m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { d = std::max(d, std::fabs(a[i] - b[i])); m = std::max(m, std::fabs(b[i])); }
    return d / m;
}

// ---- one problem on one decomposition ------------------------------------------------------------------------
struct Mono
{
    std::vector<double> x;               // PCG solution
    std::vector<double> xGs;             // symGS solution
    std::vector<double> Ax0;             // A * smooth field
    std::vector<double> r0;              // b - A * smooth field (lduMatrix::residual)
    scalar nf0 = 0;                      // normFactor of the smooth field
    scalar dot = 0, sumMag = 0, vmin = 0, vmax = 0, vmean = 0, maxRatio = 0;
    int pcgIters = 0, gsSweeps = 0;
    std::vector<scalar> hist;
};

Mono monolithic(const Problem& p, scalar tol, scalar gsTol)
{
    ProcComm* saved = ProcComm::current();
    ProcComm::setCurrent(nullptr);                                         // the single-GPU path, untouched
    const int N = p.nx*p.ny*p.nz;
    const SyntheticPartition P1 = partitionBox(p.nx, p.ny, p.nz, 1, std::vector<int>(N, 0));
    const LocalSystem s = assemble(p, P1, 0);
    DeviceSystem S(s, nullptr, nullptr, p);
    Mono m;
    DeviceBuffer<scalar> psi(s.x0), Ax, r;
    deviceAmul(S.view, psi, Ax, true);
    deviceResidual(S.view, psi, S.b, r, true);
    m.Ax0 = Ax.host();
    m.r0 = r.host();
    m.nf0 = deviceNormFactor(S.view, psi, S.b, deviceOnes(N));
    m.dot = deviceDot(psi, S.b);
    m.sumMag = deviceSumMag(psi);
    DeviceBuffer<scalar> w(std::vector<scalar>(N, 1.0));
    DeviceBuffer<scalar> mm(3);
    deviceMinMaxMeanInto(psi, mm.data());
    const std::vector<scalar> mmh = mm.host();
    m.vmin = mmh[0]; m.vmax = mmh[1]; m.vmean = mmh[2];
    m.maxRatio = deviceMaxRatio(S.M.diag, deviceOnes(N));
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = normFactorOf(S, x, N);
        const DeviceSolverPerf pf = deviceAMGPCG(S.view, S.amg, S.b, x, nf, tol, 0.0, 2000);
        const std::vector<scalar> xh = x.host();
        m.x.assign(xh.begin(), xh.end());
        m.pcgIters = pf.nIterations;
    }
    {
        DeviceBuffer<scalar> x(std::vector<scalar>(N, 0.0));
        const scalar nf = normFactorOf(S, x, N);
        DeviceSolverPerf pf;
        deviceSymGaussSeidel(S.view, S.b, x, nf, gsTol, 0.0, 20000, &pf);
        const std::vector<scalar> xh = x.host();
        m.xGs.assign(xh.begin(), xh.end());
        m.gsSweeps = pf.nIterations;
    }
    ProcComm::setCurrent(saved);
    return m;
}

void runCase(const Problem& p, const std::string& partName, const std::vector<int>& owner, const Mono& mono,
             scalar tol, scalar gsTol, bool verbose)
{
    const int np = g_comm->size(), rank = g_comm->rank();
    const int N = p.nx*p.ny*p.nz;
    const SyntheticPartition P = partitionBox(p.nx, p.ny, p.nz, np, owner);
    const LocalSystem s = assemble(p, P, rank);
    std::unique_ptr<HaloPlan> plan = g_comm->makePlan(P.descs[rank]);
    DeviceSystem S(s, g_comm, plan.get(), p);
    const std::string tag = std::string(p.name) + " " + partName + " np=" + std::to_string(np);

    // (a) the hooks -------------------------------------------------------------------------------------------
    {
        DeviceBuffer<scalar> psi(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
        psi.copyFrom(s.x0);
        DeviceBuffer<scalar> Ax, r;
        deviceAmul(S.view, psi, Ax, true);
        deviceResidual(S.view, psi, S.b, r, true);
        const std::vector<double> Axg = gatherToGlobal(s, Ax.host(), N), rg = gatherToGlobal(s, r.host(), N);
        check(maxRelDiff(Axg, mono.Ax0) < 1e-13, tag + ": distributed A*x equals the monolithic product");
        check(maxRelDiff(rg, mono.r0) < 1e-12, tag + ": distributed residual equals the monolithic residual");
        const scalar nf = normFactorOf(S, psi, s.nC);
        check(std::fabs(nf - mono.nf0) <= 1e-12*mono.nf0, tag + ": global normFactor " + std::to_string(nf) + " vs " + std::to_string(mono.nf0));
        check(sameOnAllRanks(nf), tag + ": normFactor identical on every rank");
        const scalar dot = deviceDot(psi, S.b), sm = deviceSumMag(psi);
        check(std::fabs(dot - mono.dot) <= 1e-12*std::fabs(mono.dot) && std::fabs(sm - mono.sumMag) <= 1e-12*mono.sumMag, tag + ": global dot / sumMag");
        check(sameOnAllRanks({dot, sm}), tag + ": dot and sumMag identical on every rank");
        DeviceBuffer<scalar> dd(2);
        deviceDotInto(psi, S.b, dd.data());
        deviceSumMagInto(psi, dd.data() + 1);
        const std::vector<scalar> ddh = dd.host();
        check(ddh[0] == dot && ddh[1] == sm, tag + ": device-resident reductions equal the host-returning ones");
        DeviceBuffer<scalar> mm(3);
        deviceMinMaxMeanInto(psi, mm.data());
        const std::vector<scalar> mmh = mm.host();
        check(mmh[0] == mono.vmin && mmh[1] == mono.vmax && std::fabs(mmh[2] - mono.vmean) <= 1e-13, tag + ": global min / max / mean");
        const scalar mr = deviceMaxRatio(S.M.diag, deviceOnes(s.nC));   // a ratio >= 0, as the Courant number it was written for
        check(mr == mono.maxRatio, tag + ": global max ratio " + std::to_string(mr) + " vs " + std::to_string(mono.maxRatio));
        if (np > 1)
        {
            scalar localDot;
            { LocalReductions local; localDot = deviceDot(psi, S.b); }
            const std::vector<double> ld = {localDot};
            double sum = localDot;
            g_comm->allReduceHost(&sum, 1, ReduceOp::Sum);
            check(sum == dot, tag + ": a LocalReductions scope returns the rank's own value (their rank-order sum is the global one)");
            check(deviceReductionsGlobal(), tag + ": reductions are global outside the scope");
            { LocalReductions local; check(!deviceReductionsGlobal(), tag + ": ...and local inside it"); }
        }
    }

    // (b) AMG-PCG ---------------------------------------------------------------------------------------------
    const Solve fused = pcgSolve(S, s, tol, 0.0, 2000, false);
    const Solve plain = pcgSolve(S, s, tol, 0.0, 2000, true);
    {
        const std::vector<double> xg = gatherToGlobal(s, fused.x, N);
        const double diff = maxRelDiff(xg, mono.x);
        check(fused.perf.finalResidual < tol, tag + ": AMG-PCG converged");
        check(diff < 1e-6, tag + ": AMG-PCG solution vs monolithic, rel diff " + std::to_string(diff));
        check(sameOnAllRanks({double(fused.perf.nIterations), double(plain.perf.nIterations)}), tag + ": every rank takes the same iteration count");
        std::vector<double> h(fused.hist.begin(), fused.hist.end());
        check(sameOnAllRanks(h), tag + ": every rank has the same residual history, bit for bit");
        {
            std::size_t firstDiff = fused.hist.size();
            for (std::size_t i = 0; i < std::min(fused.hist.size(), plain.hist.size()); ++i)
                if (fused.hist[i] != plain.hist[i]) { firstDiff = i; break; }
            double xd = 0;
            for (std::size_t i = 0; i < fused.x.size(); ++i) xd = std::max(xd, std::fabs(fused.x[i] - plain.x[i]));
            if (firstDiff < fused.hist.size() || xd != 0 || fused.perf.nIterations != plain.perf.nIterations)
                std::printf("  rank %d: fused vs plain: iters %d/%d, first history difference at %zu, max |dx| %.3e\n", rank,
                            fused.perf.nIterations, plain.perf.nIterations, firstDiff, xd);
        }
        check(fused.perf.nIterations == plain.perf.nIterations
              && fused.x.size() == plain.x.size()
              && std::memcmp(fused.x.data(), plain.x.data(), fused.x.size()*sizeof(scalar)) == 0
              && fused.hist.size() == plain.hist.size()
              && std::memcmp(fused.hist.data(), plain.hist.data(), fused.hist.size()*sizeof(scalar)) == 0,
              tag + ": the fused two-collective PCG is bit-identical to the plain loop");
        if (verbose && rank == 0)
            std::printf("  %-34s PCG iters %3d (monolithic %3d)  growth %.2fx  final %.2e  x diff %.2e\n", tag.c_str(),
                        fused.perf.nIterations, mono.pcgIters, double(fused.perf.nIterations)/mono.pcgIters,
                        fused.perf.finalResidual, diff);
    }

    // collectives and exchanges per iteration (reductions.cu / device_spmv.cu counters, np > 1 only)
    if (np > 1)
    {
        ProcHookCounts before = procHookCounts();
        const Solve a = pcgSolve(S, s, 1e-30, 0.0, 12, false);
        const ProcHookCounts mid = procHookCounts();
        const Solve b3 = pcgSolve(S, s, 1e-30, 0.0, 24, false);
        const ProcHookCounts after = procHookCounts();
        const double dRed = double((after.reductions - mid.reductions) - (mid.reductions - before.reductions)) / 12.0;
        const double dEx = double((after.exchanges - mid.exchanges) - (mid.exchanges - before.exchanges)) / 12.0;
        (void)a; (void)b3;
        if (verbose && rank == 0) std::printf("  %-34s per PCG iteration: %.2f collectives, %.2f halo exchanges (fused)\n", tag.c_str(), dRed, dEx);
        check(dRed == 2.0 && dEx == 1.0, tag + ": the fused PCG iteration is 2 collectives + 1 halo exchange");
        ProcHookCounts b0 = procHookCounts();
        const Solve u1 = pcgSolve(S, s, 1e-30, 0.0, 12, true);
        const ProcHookCounts b1 = procHookCounts();
        const Solve u2 = pcgSolve(S, s, 1e-30, 0.0, 24, true);
        const ProcHookCounts b2 = procHookCounts();
        const double uRed = double((b2.reductions - b1.reductions) - (b1.reductions - b0.reductions)) / 12.0;
        (void)u1; (void)u2;
        if (verbose && rank == 0) std::printf("  %-34s per PCG iteration: %.2f collectives (plain loop)\n", tag.c_str(), uRed);
        check(uRed == 3.0, tag + ": the plain PCG iteration is 3 collectives");
    }

    // (c) symGaussSeidel with a frozen halo ---------------------------------------------------------------------
    {
        const Solve g = gsSolve(S, s, gsTol, 20000);
        const std::vector<double> xg = gatherToGlobal(s, g.x, N);
        const double diff = maxRelDiff(xg, mono.xGs);
        check(g.perf.finalResidual < gsTol, tag + ": symGaussSeidel converged");
        check(diff < 2e-6, tag + ": symGaussSeidel solution vs monolithic, rel diff " + std::to_string(diff));
        check(sameOnAllRanks({double(g.perf.nIterations), g.perf.finalResidual, g.perf.initialResidual}), tag + ": every rank takes the same sweep count and reports the same residuals");
        // and it is the same solution the PCG found
        check(maxRelDiff(xg, gatherToGlobal(s, fused.x, N)) < 5e-6, tag + ": symGaussSeidel and AMG-PCG agree");
        if (verbose && rank == 0)
            std::printf("  %-34s symGS sweeps %4d (monolithic %4d)  growth %.2fx  x diff %.2e\n", tag.c_str(), g.perf.nIterations,
                        mono.gsSweeps, double(g.perf.nIterations)/mono.gsSweeps, diff);
    }
    // (f) the CUDA-graph loops on the NCCL backend (every rank captures its own graph, the collectives are in the WHILE
    // body) against the host-driven loops: the same iteration/sweep count, residuals, history and solution, bit for bit
    if (np > 1 && g_comm->backend() == ProcComm::Backend::Nccl && braeCondGraphsAvailable())
    {
        auto bitsEq = [](const std::vector<scalar>& a, const std::vector<scalar>& b)
        { return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0); };
        auto sameBits = [&](const Solve& a, const Solve& b)
        {
            return a.perf.nIterations == b.perf.nIterations
                && std::memcmp(&a.perf.initialResidual, &b.perf.initialResidual, sizeof(scalar)) == 0
                && std::memcmp(&a.perf.finalResidual, &b.perf.finalResidual, sizeof(scalar)) == 0
                && bitsEq(a.x, b.x) && bitsEq(a.hist, b.hist);
        };
        struct PcgCfg { const char* name; scalar tol, rel; int maxIter, minIter; };
        const PcgCfg pcgCfgs[] =
        {
            {"tol 1e-10",                       1e-10, 0.0,  2000, 0},
            {"tol 1e-10 (graph replayed)",      1e-10, 0.0,  2000, 0},
            {"relTol 0.05",                     1e-30, 0.05, 2000, 0},
            {"maxIter cap 7",                   1e-30, 0.0,  7,    0},
            {"converged at the start",          1e10,  0.0,  2000, 0},
            {"minIter 4 on a converged start",  1e10,  0.0,  2000, 4},
            {"tol 1e-10 after the others",      1e-10, 0.0,  2000, 0},
        };
        for (const PcgCfg& cfg : pcgCfgs)
        {
            const Solve host = pcgSolve(S, s, cfg.tol, cfg.rel, cfg.maxIter, false, cfg.minIter, false);
            const Solve gr = pcgSolve(S, s, cfg.tol, cfg.rel, cfg.maxIter, false, cfg.minIter, true);
            check(sameBits(host, gr), tag + ": graph AMG-PCG [" + cfg.name + "] is bit-identical to the host-driven loop (iters "
                  + std::to_string(gr.perf.nIterations) + "/" + std::to_string(host.perf.nIterations) + ", final "
                  + std::to_string(gr.perf.finalResidual) + "/" + std::to_string(host.perf.finalResidual) + ")");
            std::vector<double> h(gr.hist.begin(), gr.hist.end());
            check(sameOnAllRanks(h) && sameOnAllRanks(double(gr.perf.nIterations)),
                  tag + ": graph AMG-PCG [" + cfg.name + "]: every rank, same iteration count and history");
            if (verbose && rank == 0 && cfg.maxIter == 2000 && cfg.minIter == 0 && cfg.rel == 0.0 && cfg.tol == 1e-10)
                std::printf("  %-34s graph PCG %d iterations == host-driven %d, history of %zu residuals identical\n", tag.c_str(),
                            gr.perf.nIterations, host.perf.nIterations, gr.hist.size());
        }
        {   // the collectives are INSIDE the graph: extra iterations add no host-issued collective or exchange
            const ProcHookCounts c0 = procHookCounts();
            pcgSolve(S, s, 1e-30, 0.0, 12, false, 0, true);
            const ProcHookCounts c1 = procHookCounts();
            pcgSolve(S, s, 1e-30, 0.0, 24, false, 0, true);
            const ProcHookCounts c2 = procHookCounts();
            check((c2.reductions - c1.reductions) == (c1.reductions - c0.reductions)
                  && (c2.exchanges - c1.exchanges) == (c1.exchanges - c0.exchanges),
                  tag + ": graph AMG-PCG: 12 more iterations add no host-issued collective or halo exchange");
        }
        struct GsCfg { const char* name; scalar tol, rel; int maxIter, minIter, nSweeps; };
        const GsCfg gsCfgs[] =
        {
            {"tol 1e-10",                    gsTol, 0.0,  20000, 0, 1},
            {"tol 1e-10 (graph replayed)",   gsTol, 0.0,  20000, 0, 1},
            {"relTol 0.1",                   1e-30, 0.1,  2000,  0, 1},
            {"maxIter cap 5",                1e-30, 0.0,  5,     0, 1},
            {"nSweeps 2",                    1e-8,  0.0,  5000,  0, 2},
            {"converged at the start",       1e10,  0.0,  100,   0, 1},
            {"minIter 4 on a converged start", 1e10, 0.0, 100,   4, 1},
        };
        for (const GsCfg& cfg : gsCfgs)
        {
            const Solve host = gsSolve(S, s, cfg.tol, cfg.maxIter, cfg.rel, cfg.minIter, cfg.nSweeps, false);
            const Solve gr = gsSolve(S, s, cfg.tol, cfg.maxIter, cfg.rel, cfg.minIter, cfg.nSweeps, true);
            check(sameBits(host, gr), tag + ": graph symGaussSeidel [" + cfg.name + "] is bit-identical to the host-driven loop (sweeps "
                  + std::to_string(gr.perf.nIterations) + "/" + std::to_string(host.perf.nIterations) + ", final "
                  + std::to_string(gr.perf.finalResidual) + "/" + std::to_string(host.perf.finalResidual) + ")");
            check(sameOnAllRanks({double(gr.perf.nIterations), gr.perf.finalResidual, gr.perf.initialResidual}),
                  tag + ": graph symGaussSeidel [" + cfg.name + "]: every rank, same sweep count and residuals");
            if (verbose && rank == 0 && cfg.name == gsCfgs[0].name)
                std::printf("  %-34s graph symGS %d sweeps == host-driven %d\n", tag.c_str(), gr.perf.nIterations, host.perf.nIterations);
        }
    }
    // (e) the other Krylov loops inherit the hooks: Jacobi-PCG, and BiCGStab with the block-local AMG V-cycle
    {
        DeviceBuffer<scalar> x(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
        x.copyFrom(std::vector<scalar>(s.nC, 0.0));
        const scalar nf = normFactorOf(S, x, s.nC);
        const DeviceSolverPerf pj = deviceJacobiPCG(S.view, S.b, x, nf, 1e-10, 0.0, 20000);
        const double dj = maxRelDiff(gatherToGlobal(s, x.host(), N), mono.x);
        check(pj.finalResidual < 1e-10 && dj < 1e-5, tag + ": Jacobi-PCG on the decomposed matrix, diff " + std::to_string(dj));
        check(sameOnAllRanks(double(pj.nIterations)), tag + ": Jacobi-PCG: the same iteration count on every rank");
        DeviceBuffer<scalar> xb(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
        xb.copyFrom(std::vector<scalar>(s.nC, 0.0));
        const DeviceSolverPerf pb = deviceJacobiBiCGStab(S.view, S.b, xb, nf, 1e-10, 0.0, 5000, 1, 0, nullptr, &S.amg, 1);
        const double db = maxRelDiff(gatherToGlobal(s, xb.host(), N), mono.x);
        check(pb.finalResidual < 1e-10 && db < 1e-5, tag + ": BiCGStab (AMG) on the decomposed matrix, diff " + std::to_string(db));
        check(sameOnAllRanks(double(pb.nIterations)), tag + ": BiCGStab: the same iteration count on every rank");
        if (verbose && rank == 0)
            std::printf("  %-34s Jacobi-PCG %d iters, BiCGStab(AMG) %d iters\n", tag.c_str(), pj.nIterations, pb.nIterations);
    }
    g_comm->barrier();
}

// ---- microbenchmark: one distributed PCG iteration at ~310k cells per rank --------------------------------------
void bench()
{
    const int np = g_comm->size(), rank = g_comm->rank();
    // 68^3 = 314,432 cells per rank; the rank grid px*py*pz = np
    const int per = 68;
    int px = 1, py = 1, pz = 1;
    if (np == 2) px = 2; else if (np == 4) { px = 2; py = 2; } else if (np == 8) { px = 2; py = 2; pz = 2; } else px = np;
    Problem p{"bench", per*px, per*py, per*pz, 1.0, 4.0, 0.25, 0.05};
    const std::vector<int> owner = blockOwner(p.nx, p.ny, p.nz, px, py, pz);
    const SyntheticPartition P = partitionBox(p.nx, p.ny, p.nz, np, owner);
    const LocalSystem s = assemble(p, P, rank);
    std::unique_ptr<HaloPlan> plan = g_comm->makePlan(P.descs[rank]);
    DeviceSystem S(s, g_comm, plan.get(), p);
    // host-driven loops (fused, plain) and, on the NCCL backend, the whole-loop graph; the timed region is the solve
    // alone (no history hook, no host copy of the solution); best of 3 solves of 60 iterations after a warm-up solve
    struct Mode { const char* name; bool unfused; bool graph; };
    std::vector<Mode> modes = {{"host fused ", false, false}, {"host plain ", true, false}};
    if (g_comm->backend() == ProcComm::Backend::Nccl && np > 1 && braeCondGraphsAvailable()) modes.push_back({"graph fused", false, true});
    for (const Mode& m : modes)
    {
        DeviceBuffer<scalar> x(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
        x.copyFrom(std::vector<scalar>(s.nC, 0.0));
        const scalar nf = normFactorOf(S, x, s.nC);
        if (m.unfused) setenv("BRAE_PROC_PCG_UNFUSED", "1", 1); else unsetenv("BRAE_PROC_PCG_UNFUSED");
        setenv("BRAE_PROC_GRAPH", m.graph ? "1" : "0", 1);
        const int iters = 60;
        double best = 1e300;
        int n = 0;
        for (int rep = 0; rep < 4; ++rep)                                // rep 0: warm-up (graph capture, caches)
        {
            x.copyFrom(std::vector<scalar>(s.nC, 0.0));
            g_comm->barrier();
            const auto t0 = std::chrono::steady_clock::now();
            const DeviceSolverPerf pf = deviceAMGPCG(S.view, S.amg, S.b, x, nf, 1e-30, 0.0, iters);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            n = pf.nIterations;
            if (rep > 0) best = std::min(best, ms);
        }
        unsetenv("BRAE_PROC_PCG_UNFUSED");
        g_comm->barrier();
        double worst = best;
        g_comm->allReduceHost(&worst, 1, ReduceOp::Max);
        if (rank == 0)
            std::printf("bench np=%d backend=%s %s: %d cells/rank, %d PCG iterations: %.3f ms/iteration (slowest rank; all ranks "
                        "sharing ONE GPU, relative numbers only)\n", np, ProcComm::backendName(g_comm->backend()), m.name, s.nC, n,
                        worst/n);
    }
}

} // namespace

int main(int argc, char** argv)
{
    int status = 0;
    try
    {
        Pstream::init(argc, argv);
        std::vector<std::unique_ptr<ProcComm>> comms = ProcComm::create();          // BRAE_COMM = mpi (default here) | nccl | auto
        std::unique_ptr<ProcComm> comm = std::move(comms.at(0));
        g_comm = comm.get();
        ProcComm::setCurrent(comm.get());
        const int np = comm->size();
        if (argc > 1 && std::string(argv[1]) == "bench")
        {
            bench();
        }
        else
        {
            const bool verbose = true;
            const scalar tol = 1e-10, gsTol = 1e-10;
            const Problem problems[] =
            {
                {"poisson 40^3 ", 40, 40, 40, 1.0, 1.0, 1.0, 0.05},
                {"aniso 40x32x24", 40, 32, 24, 1.0, 4.0, 0.25, 0.05},
            };
            for (const Problem& p : problems)
            {
                const int nx = p.nx, ny = p.ny, nz = p.nz;
                const Mono mono = monolithic(p, tol, gsTol);
                if (comm->isMaster())
                    std::printf("%s monolithic: PCG %d iterations, symGS %d sweeps\n", p.name, mono.pcgIters, mono.gsSweeps);
                int px = 1, py = 1, pz = 1;
                if (np == 2) px = 2; else if (np == 4) { px = 2; py = 2; } else if (np == 8) { px = 2; py = 2; pz = 2; } else px = np;
                runCase(p, "block", blockOwner(nx, ny, nz, px, py, pz), mono, tol, gsTol, verbose);
                if (np > 1)
                    runCase(p, "scattered", scatteredOwner(nx, ny, nz, np, 4), mono, tol, gsTol, verbose);
            }
        }
        const int total = static_cast<int>(comm->allReduce(static_cast<double>(g_fail), ReduceOp::Sum));
        if (comm->isMaster()) std::printf("test_dist_solvers np=%d: %s (%d failed checks)\n", np, total ? "FAIL" : "PASS", total);
        status = total ? 1 : 0;
        ProcComm::setCurrent(nullptr);
        comm.reset();
        Pstream::finalize();
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL: exception %s\n", e.what());
        std::fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return status;
}
