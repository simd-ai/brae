// Processor-face pass grouping (review finding 13): the grouped launch must be BIT-IDENTICAL to the old
// full-face/filter launch. Self-contained (no dist_solvers fixture): one rank builds a HaloPlan whose cells own
// 1..3 processor faces, the interface groups the faces stably by pass, and deviceAmul/deviceResidual/
// deviceProcFrozenRhs are compared bit for bit between the grouped path and BRAE_PROC_FULL_FACE_LAUNCH=1.
#include "proc_comm.cuh"
#include "device_proc_interface.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <cstring>
#include <vector>

using namespace brae;

namespace {

void setenvInt(const char* name, bool on)
{
    if (on) setenv(name, "1", 1);
    else unsetenv(name);
}

std::vector<double> amul(DeviceLduView& A, const DeviceBuffer<scalar>& x)
{
    DeviceBuffer<scalar> y;
    deviceAmul(A, x, y);
    return y.host();
}

} // namespace

void runFixture(ProcComm& comm, int rank);

int main()
{
    auto comms = ProcComm::createLoopback(2);
    // DevicePool is documented single-threaded (device_buffer.cuh); this fixture runs one solver rank per host
    // thread, so it must use raw cudaMalloc rather than the shared free list.
    setenv("BRAE_NO_DEVICE_POOL", "1", 1);
    std::vector<std::thread> ranks;
    std::vector<int> rankBad(2, 0);
    for (int r = 0; r < 2; ++r)
        ranks.emplace_back([&, r] {
            comms[r]->bindThread();
            try { runFixture(*comms[r], r); }
            catch (const std::exception& e) { ++rankBad[r]; std::printf("rank %d: %s\n", r, e.what()); }
        });
    for (auto& t : ranks) t.join();
    return (rankBad[0] || rankBad[1]) ? 1 : 0;
}

// The SPMD fixture: BOTH loopback ranks construct the same (mirrored) plan and exchange halos, so the collectives
// have their partners. Only rank 0 prints/compares.
void runFixture(ProcComm& comm, int rank)
{
    // 6 owned cells, 8 processor faces; cell 0 owns faces 0,1,2 (passes 0,1,2), cell 1 owns 3,4 (0,1), etc.
    HaloPlanDesc d;
    d.nCells = 6;
    d.nHalo = 8;
    d.nProcFaces = 8;
    d.nbrs.resize(1);
    d.nbrs[0].rank = 1 - rank;
    d.nbrs[0].sendOffset = 0;
    d.nbrs[0].sendCount = 8;
    d.nbrs[0].recvStart = 0;
    d.nbrs[0].recvCount = 8;
    d.nbrs[0].faceStart = 0;
    d.nbrs[0].faceCount = 8;
    d.sendCells = {0, 1, 2, 3, 4, 5, 0, 1};
    d.faceCells = {0, 0, 0, 1, 1, 2, 3, 4};
    d.faceToSlot = {0, 1, 2, 3, 4, 5, 6, 7};
    auto plan = comm.makePlan(d);
    DeviceProcInterface ifc(comm, *plan);
    std::vector<scalar> coeff(8);
    for (int i = 0; i < 8; ++i) coeff[i] = 0.25 + 0.125*i;
    ifc.setCoeff(coeff);

    const int nC = 6;
    std::vector<scalar> diag(nC, 2.0), upper, lower, b(nC, 1.0), x(nC + 8);
    std::vector<label> owner, nei;
    for (int i = 0; i + 1 < nC; ++i)
    {
        owner.push_back(i);
        nei.push_back(i + 1);
        upper.push_back(-0.5);
        lower.push_back(-0.5);
    }
    for (int i = 0; i < nC + 8; ++i) x[i] = 1.0 + 0.01*i;
    DeviceLduMatrix M = buildDeviceLdu(diag, upper, lower, owner, nei, nC);
    DeviceLduView A = M.view();
    A.proc = ifc.view();
    DeviceBuffer<scalar> dx(nC, 8);
    dx.copyFrom(x);
    DeviceBuffer<scalar> db(nC);
    db.copyFrom(b);

    setenvInt("BRAE_PROC_FULL_FACE_LAUNCH", false);
    const std::vector<double> grouped = amul(A, dx);
    DeviceBuffer<scalar> rg;
    deviceResidual(A, dx, db, rg);
    const std::vector<double> resGrouped = rg.host();
    DeviceBuffer<scalar> bg;
    deviceProcFrozenRhs(A, dx, db, bg);
    const std::vector<double> rhsGrouped = bg.host();

    setenvInt("BRAE_PROC_FULL_FACE_LAUNCH", true);
    const std::vector<double> full = amul(A, dx);
    DeviceBuffer<scalar> rf;
    deviceResidual(A, dx, db, rf);
    const std::vector<double> resFull = rf.host();
    DeviceBuffer<scalar> bf;
    deviceProcFrozenRhs(A, dx, db, bf);
    const std::vector<double> rhsFull = bf.host();
    setenvInt("BRAE_PROC_FULL_FACE_LAUNCH", false);

    int bad = 0;
    auto compare = [&](const char* what, const std::vector<double>& a, const std::vector<double>& b2) {
        if (a.size() != b2.size() || (a.size() && std::memcmp(a.data(), b2.data(), a.size()*sizeof(double)) != 0))
        {
            ++bad;
            std::printf("FAIL %s: grouped launch is not bit-identical to the full-face launch\n", what);
            for (std::size_t i = 0; i < a.size() && i < b2.size(); ++i)
                if (a[i] != b2[i]) std::printf("  [%zu] grouped %.17g full %.17g\n", i, a[i], b2[i]);
        }
    };
    compare("deviceAmul", grouped, full);
    compare("deviceResidual", resGrouped, resFull);
    compare("deviceProcFrozenRhs", rhsGrouped, rhsFull);
    // The pass ranges themselves: 3 passes, faces grouped as {0,3,5,6,7}, {1,4}, {2}.
    const DeviceProcView v = ifc.view();
    if (v.nRanks != 3 || v.passFirst[0] != 0 || v.passFirst[1] != 5 || v.passFirst[2] != 7 || v.passFirst[3] != 8)
    {
        ++bad;
        std::printf("FAIL: pass ranges nRanks=%d starts=%d,%d,%d,%d (expected 3: 0,5,7,8)\n",
                    v.nRanks, v.passFirst[0], v.passFirst[1], v.passFirst[2], v.passFirst[3]);
    }
    // Loopback plan-lifetime rule: destroy the plan only after every rank has finished its last exchange on it.
    comm.barrier();
    if (bad) throw std::runtime_error("pass-grouping comparison failed");
    std::printf("PASS proc pass grouping: AMUL, residual and frozen RHS bit-identical (8 faces, 3 passes)\n");
}
