// brae's distributed solve of the p_rgh matrix OpenFOAM v2606 -parallel handed to its own solver (tools/mgpu_oracle,
// TIGHT=1 run, dump oPrghEqnc<k>n<j>), against OpenFOAM's per-rank solution and its reported initial residual.
//
//   tools/mgpu_oracle/run_oracle.sh open 4 scotch 1   (TIGHT=1)
//   tools/mgpu_oracle/export_prgh_matrix.py <run>/par_d1 <dir> <corrector> 1
//   mpirun --oversubscribe -np 4 test_dist_solvers_oracle <dir> [<monolithic dir>]
//
// The exported rank files hold the local lduMatrix as the solver sees it (interface coefficients and the diagonal share of the
// processor patches included), the p_rgh before the solve (the initial guess) and OpenFOAM's p_rgh after it. Checked:
//   * the initial residual brae computes with the GLOBAL normFactor and global sum|r| equals the one OpenFOAM printed
//   * brae's AMG-PCG (per-rank AMG hierarchy, block-local V-cycle, halo-exchanged products) reaches the tolerance with the
//     same iteration count on every rank, and its solution equals OpenFOAM's per rank
//   * the iteration count against the same system solved monolithically (the serial dump), when given
//   * symGaussSeidel with the frozen halo reaches the same solution
// An exit of 77 means "dump not found" (ctest SKIP).
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
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

struct Patch
{
    int nbr = -1;
    std::vector<int> faceCells, nbrFaceCells;
    std::vector<double> bou;
};

struct RankData
{
    int nC = 0, nF = 0;
    std::vector<double> diag, source, upper, lower, psi0, psiRef;
    std::vector<int> owner, nei;
    std::vector<Patch> patches;
};

template <class T> void rd(std::ifstream& f, std::vector<T>& v, std::size_t n)
{
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n*sizeof(T)));
    if (!f) throw std::runtime_error("short read");
}

bool load(const std::string& dir, int rank, RankData& d)
{
    std::ifstream f(dir + "/rank" + std::to_string(rank) + ".bin", std::ios::binary);
    if (!f) return false;
    int hdr[5];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (hdr[0] != 0x70524748) throw std::runtime_error("bad magic");
    d.nC = hdr[1]; d.nF = hdr[2];
    const int nP = hdr[3];
    rd(f, d.diag, d.nC); rd(f, d.source, d.nC); rd(f, d.upper, d.nF); rd(f, d.lower, d.nF);
    rd(f, d.owner, d.nF); rd(f, d.nei, d.nF); rd(f, d.psi0, d.nC); rd(f, d.psiRef, d.nC);
    for (int p = 0; p < nP; ++p)
    {
        Patch pt;
        int h2[2];
        f.read(reinterpret_cast<char*>(h2), sizeof(h2));
        pt.nbr = h2[0];
        rd(f, pt.faceCells, h2[1]); rd(f, pt.nbrFaceCells, h2[1]); rd(f, pt.bou, h2[1]);
        d.patches.push_back(std::move(pt));
    }
    return true;
}

struct OfReport { double initial = -1, final_ = -1; int iters = -1; int np = 0; long long cells = 0; };

OfReport readReport(const std::string& dir)
{
    OfReport r;
    std::ifstream f(dir + "/solver.txt");
    std::string l1, l2;
    std::getline(f, l1); std::getline(f, l2);
    std::istringstream is(l1);
    is >> r.np >> r.cells;
    std::smatch m;
    if (std::regex_search(l2, m, std::regex("Initial residual = ([0-9.eE+-]+), Final residual = ([0-9.eE+-]+), No Iterations ([0-9]+)")))
    {
        r.initial = std::stod(m[1]); r.final_ = std::stod(m[2]); r.iters = std::stoi(m[3]);
    }
    return r;
}

struct System
{
    int nC = 0, nHalo = 0;
    DeviceLduMatrix M;
    DeviceBuffer<scalar> b;
    std::unique_ptr<HaloPlan> plan;
    std::unique_ptr<DeviceProcInterface> ifc;
    DeviceLduView view;
    AMGData amg;
    double sign = 1;
};

std::unique_ptr<System> build(RankData d, ProcComm* comm)
{
    auto s = std::make_unique<System>();
    s->nC = d.nC;
    // OpenFOAM's pressure matrix is negative definite (diag < 0): solve the same system as -A x = -b, which is what the
    // positive-definite solvers expect (the solution and the normalised residual are unchanged)
    double dsum = 0;
    for (double v : d.diag) dsum += v;
    if (dsum < 0)
    {
        s->sign = -1;
        for (auto* v : {&d.diag, &d.source, &d.upper, &d.lower}) for (double& x : *v) x = -x;
        for (Patch& p : d.patches) for (double& x : p.bou) x = -x;
    }
    std::vector<scalar> faceW(d.nF);
    for (int f = 0; f < d.nF; ++f) faceW[f] = std::fabs(d.upper[f]);
    s->M = buildDeviceLdu(d.diag, d.upper, d.lower, d.owner, d.nei, d.nC);
    s->b.copyFrom(d.source);
    s->view = s->M.view();
    if (comm)
    {
        std::vector<ProcessorPatchDesc> pp;
        std::vector<scalar> coeff;
        for (const Patch& p : d.patches)
        {
            ProcessorPatchDesc pd;
            pd.nbrRank = p.nbr; pd.faceCells = p.faceCells; pd.nbrFaceCells = p.nbrFaceCells;
            pp.push_back(std::move(pd));
            for (double x : p.bou) coeff.push_back(-x);                 // Apsi -= bou*x_nbr  <=>  Apsi += (-bou)*x_nbr
        }
        s->plan = comm->makePlan(buildHaloPlanFromProcessorPatches(d.nC, pp));
        s->nHalo = s->plan->nHalo();
        s->ifc = std::make_unique<DeviceProcInterface>(*comm, *s->plan);
        s->ifc->setCoeff(coeff);
        s->view.proc = s->ifc->view();
    }
    s->amg = buildAMG(d.owner, d.nei, faceW, d.nC);
    amgGalerkin(s->amg, s->M.diag, s->M.upper, s->M.lower);
    return s;
}

struct Result { DeviceSolverPerf perf; std::vector<scalar> x; };

Result solvePcg(System& s, const std::vector<double>& x0, scalar tol, bool graph = false)
{
    DeviceBuffer<scalar> x(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
    setenv("BRAE_PROC_GRAPH", graph ? "1" : "0", 1);
    x.copyFrom(x0);
    const scalar nf = deviceNormFactor(s.view, x, s.b, deviceOnes(s.nC));
    Result r;
    r.perf = deviceAMGPCG(s.view, s.amg, s.b, x, nf, tol, 0.0, 5000);
    r.x = x.host();
    return r;
}

Result solveGs(System& s, const std::vector<double>& x0, scalar tol, bool graph = false)
{
    DeviceBuffer<scalar> x(static_cast<std::size_t>(s.nC), static_cast<std::size_t>(s.nHalo));
    x.copyFrom(x0);
    setenv("BRAE_PROC_GRAPH", graph ? "1" : "0", 1);
    const scalar nf = deviceNormFactor(s.view, x, s.b, deviceOnes(s.nC));
    Result r;
    deviceSymGaussSeidel(s.view, s.b, x, nf, tol, 0.0, 200000, &r.perf);
    r.x = x.host();
    return r;
}

double maxRel(const std::vector<scalar>& a, const std::vector<double>& ref)
{
    // a is the sign-adjusted solve of -A x = -b: the same x
    double d = 0, m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { d = std::max(d, std::fabs(a[i] - ref[i])); m = std::max(m, std::fabs(ref[i])); }
    return d / m;
}

double gmax(double v) { g_comm->allReduceHost(&v, 1, ReduceOp::Max); return v; }

} // namespace

int main(int argc, char** argv)
{
    int status = 0;
    try
    {
        Pstream::init(argc, argv);
        std::vector<std::unique_ptr<ProcComm>> comms = ProcComm::create();        // BRAE_COMM = mpi (default) | nccl | auto
        std::unique_ptr<ProcComm> comm = std::move(comms.at(0));
        g_comm = comm.get();
        ProcComm::setCurrent(comm.get());
        if (argc < 2) { if (comm->isMaster()) std::printf("usage: test_dist_solvers_oracle <dump dir> [<monolithic dump dir>]\n"); ProcComm::setCurrent(nullptr); comm.reset(); Pstream::finalize(); return 77; }
        const std::string dir = argv[1];
        const int np = comm->size(), rank = comm->rank();
        RankData d;
        int have = load(dir, rank, d) ? 1 : 0;
        double hv = have;
        comm->allReduceHost(&hv, 1, ReduceOp::Min);
        if (hv < 1)
        {
            if (comm->isMaster()) std::printf("SKIP: no exported dump for %d ranks in %s\n", np, dir.c_str());
            // torn down as the normal end is: a communicator that outlives MPI_Finalize aborts in MPI_Test
            ProcComm::setCurrent(nullptr);
            comm.reset();
            Pstream::finalize();
            return 77;
        }
        const OfReport of = readReport(dir);
        check(of.np == np, "the dump was exported for " + std::to_string(of.np) + " ranks, this run has " + std::to_string(np));
        const std::vector<double> x0 = d.psi0, xRef = d.psiRef;
        std::unique_ptr<System> sys = build(std::move(d), comm.get());

        const scalar tol = 1e-12;
        const Result pcg = solvePcg(*sys, x0, tol);
        const double relI = std::fabs(pcg.perf.initialResidual - of.initial) / of.initial;
        const double diff = gmax(maxRel(pcg.x, xRef));
        std::vector<double> it = {double(pcg.perf.nIterations)};
        double lo = it[0], hi = it[0];
        comm->allReduceHost(&lo, 1, ReduceOp::Min); comm->allReduceHost(&hi, 1, ReduceOp::Max);
        check(lo == hi, "every rank takes the same iteration count");
        check(relI < 1e-6 || (of.initial < 1e-6 && relI < 5e-3), "initial residual " + std::to_string(pcg.perf.initialResidual)
              + " vs OpenFOAM " + std::to_string(of.initial));
        check(pcg.perf.finalResidual < tol, "AMG-PCG reached the tolerance");
        check(diff < 1e-6, "solution against OpenFOAM's, max over ranks " + std::to_string(diff));

        const Result gs = solveGs(*sys, x0, 1e-10);
        const double gdiff = gmax(maxRel(gs.x, xRef));
        check(gs.perf.finalResidual < 1e-10, "symGaussSeidel reached the tolerance");
        // the CUDA-graph loops on the NCCL backend against the host-driven ones on the OpenFOAM-dumped matrices
        if (np > 1 && comm->backend() == ProcComm::Backend::Nccl && braeCondGraphsAvailable())
        {
            const Result pg = solvePcg(*sys, x0, tol, true);
            check(pg.perf.nIterations == pcg.perf.nIterations
                  && std::memcmp(&pg.perf.finalResidual, &pcg.perf.finalResidual, sizeof(scalar)) == 0
                  && std::memcmp(&pg.perf.initialResidual, &pcg.perf.initialResidual, sizeof(scalar)) == 0
                  && pg.x.size() == pcg.x.size()
                  && std::memcmp(pg.x.data(), pcg.x.data(), pg.x.size()*sizeof(scalar)) == 0,
                  "graph AMG-PCG is bit-identical to the host-driven loop on the OpenFOAM matrix");
            const Result gg = solveGs(*sys, x0, 1e-10, true);
            check(gg.perf.nIterations == gs.perf.nIterations
                  && std::memcmp(&gg.perf.finalResidual, &gs.perf.finalResidual, sizeof(scalar)) == 0
                  && gg.x.size() == gs.x.size()
                  && std::memcmp(gg.x.data(), gs.x.data(), gg.x.size()*sizeof(scalar)) == 0,
                  "graph symGaussSeidel is bit-identical to the host-driven loop on the OpenFOAM matrix");
            if (comm->isMaster())
                std::printf("  graph vs host-driven on the OpenFOAM matrix: PCG %d iterations, symGS %d sweeps, bit-identical\n",
                            pg.perf.nIterations, gg.perf.nIterations);
        }
        check(gdiff < 1e-4, "symGaussSeidel solution against OpenFOAM's " + std::to_string(gdiff));

        // the same system, monolithic (serial dump): iteration growth with np
        int monoIters = -1, monoGs = -1;
        if (argc > 2)
        {
            ProcComm::setCurrent(nullptr);
            RankData m;
            if (load(argv[2], 0, m))
            {
                const std::vector<double> mx0 = m.psi0;
                std::unique_ptr<System> ms = build(std::move(m), nullptr);
                const Result r1 = solvePcg(*ms, mx0, tol);
                monoIters = r1.perf.nIterations;
                const Result g1 = solveGs(*ms, mx0, 1e-10);
                monoGs = g1.perf.nIterations;
            }
            ProcComm::setCurrent(comm.get());
        }
        if (comm->isMaster())
            std::printf("oracle %s: np=%d cells %lld | initial residual brae %.6e, OpenFOAM %.6e | PCG iters %d (monolithic %d), OpenFOAM GAMG %d "
                        "| final %.2e | max rel diff to OpenFOAM's p_rgh %.2e | symGS sweeps %d (monolithic %d), diff %.2e\n",
                        dir.c_str(), np, of.cells, pcg.perf.initialResidual, of.initial, pcg.perf.nIterations, monoIters, of.iters,
                        pcg.perf.finalResidual, diff, gs.perf.nIterations, monoGs, gdiff);
        const int total = static_cast<int>(comm->allReduce(static_cast<double>(g_fail), ReduceOp::Sum));
        if (comm->isMaster()) std::printf("test_dist_solvers_oracle np=%d: %s (%d failed checks)\n", np, total ? "FAIL" : "PASS", total);
        status = total ? 1 : 0;
        ProcComm::setCurrent(nullptr);
        sys.reset();
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
