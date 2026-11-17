// Global Courant numbers and time steps (review findings 4 and 6).
//
//   mpirun --oversubscribe -np N test_courant_global      N = 1,2,4,8 (MPI-host backend, one GPU is fine)
//
// (a) HOST path: ranks are given deliberately DIFFERENT local maxima and volumes; courantNo / alphaCourantNo must
//     return the global values, so setInitialDeltaT/setDeltaT produce BIT-IDENTICAL deltaT sequences on every
//     rank, and they must equal the values computed from the globally gathered data on rank 0.
// (b) ZERO-CELL rank: one rank has no cells. deviceAlphaCourantNo must still join every global reduction (zero
//     contributions) and return the combined numbers, and deviceSumMag/deviceMaxRatio on empty buffers must do
//     the same. Before the fix the empty rank returned early and the others blocked in the collectives (the
//     watchdog catches that).
// (c) np == 1: the reductions are the identity, so the single-GPU numbers are unchanged.
#include "proc_comm.cuh"
#include "time_controls.cuh"
#include "device_alpha_courant.cuh"
#include "device_blas.cuh"
#include "cf_pstream.cuh"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <thread>

using namespace brae;

namespace {

struct Watchdog
{
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::thread t;
    explicit Watchdog(int seconds)
    {
        t = std::thread([this, seconds] {
            std::unique_lock<std::mutex> lk(m);
            if (!cv.wait_for(lk, std::chrono::seconds(seconds), [&] { return done; }))
            {
                std::printf("FAIL: watchdog: a rank is stuck in a collective (zero-cell early return?)\n");
                std::fflush(stdout);
                std::_Exit(3);
            }
        });
    }
    ~Watchdog()
    {
        { std::lock_guard<std::mutex> lk(m); done = true; }
        cv.notify_all();
        t.join();
    }
};

int fails = 0;
void check(bool ok, const char* what, int rank)
{
    if (!ok) { ++fails; std::printf("FAIL rank %d: %s\n", rank, what); std::fflush(stdout); }
}

bool sameOnAll(const double& v, MPI_Comm comm)
{
    const long long bits = *reinterpret_cast<const long long*>(&v);
    std::vector<long long> all;
    int np = 0;
    MPI_Comm_size(comm, &np);
    all.resize(np);
    MPI_Allgather(&bits, 1, MPI_LONG_LONG, all.data(), 1, MPI_LONG_LONG, comm);
    for (int r = 0; r < np; ++r) if (all[r] != bits) return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int np = Pstream::nProcs();
    const int rank = Pstream::myProcNo();
    int rc = 0;
    {
        Watchdog wd(180);
        std::unique_ptr<ProcComm> comm = ProcComm::createMpiHost();
        ProcComm::setCurrent(comm.get());

        // ---- (a) non-uniform local maxima: the time-step sequence must be global and identical everywhere ----
        TimeControls tc;
        tc.adjustTimeStep = true;
        tc.maxCo = 0.5;
        tc.maxDeltaT = 0.01;
        // rank r has 4 cells; its fastest cell is r-dependent, so a rank-local Courant number would differ.
        std::vector<scalar> V(4, 1.0 + 0.25*rank);
        std::vector<scalar> sumPhi(4);
        for (int c = 0; c < 4; ++c) sumPhi[c] = (rank + 1)*0.1*(c + 1)*V[c];
        std::vector<scalar> alpha(4, rank % 2 == 0 ? 0.5 : 0.0);   // only even ranks have an interface band
        scalar dt = 0.005;
        for (int step = 0; step < 6; ++step)
        {
            const CourantNumbers co = courantNo(sumPhi, V, dt);
            const CourantNumbers aco = alphaCourantNo(sumPhi, alpha, V, dt);
            check(sameOnAll(co.CoNum, MPI_COMM_WORLD), "courantNo CoNum differs between ranks", rank);
            check(sameOnAll(co.meanCoNum, MPI_COMM_WORLD), "courantNo meanCoNum differs between ranks", rank);
            check(sameOnAll(aco.CoNum, MPI_COMM_WORLD), "alphaCourantNo CoNum differs between ranks", rank);
            if (step == 0) dt = setInitialDeltaT(dt, std::max(co.CoNum, aco.CoNum), tc);
            else dt = setDeltaT(dt, std::max(co.CoNum, aco.CoNum), tc);
            check(sameOnAll(dt, MPI_COMM_WORLD), "deltaT differs between ranks (rank-local Courant number)", rank);
            for (int c = 0; c < 4; ++c) sumPhi[c] *= 1.0 + 0.1*(step + 1);
        }
        // the oracle: the same formulas on globally gathered data, computed redundantly on every rank
        {
            std::vector<double> allPhi, allV;
            for (int r = 0; r < np; ++r)
                for (int c = 0; c < 4; ++c)
                {
                    const double v = 1.0 + 0.25*r;
                    double p = (r + 1)*0.1*(c + 1)*v;
                    for (int step = 0; step < 6; ++step) p *= 1.0 + 0.1*(step + 1);
                    allPhi.push_back(p);
                    allV.push_back(v);
                }
            double maxRatio = 0, sPhi = 0, sV = 0;
            for (std::size_t i = 0; i < allPhi.size(); ++i)
            {
                maxRatio = std::max(maxRatio, allPhi[i]/allV[i]);
                sPhi += allPhi[i];
                sV += allV[i];
            }
            const double want = 0.5*maxRatio*dt;
            const CourantNumbers co = courantNo(sumPhi, V, dt);
            check(std::fabs(co.CoNum - want) <= 1e-12*std::fabs(want), "courantNo is not the global maximum ratio", rank);
            check(std::fabs(co.meanCoNum - 0.5*(sPhi/sV)*dt) <= 1e-12*std::fabs(want), "courantNo mean is not global", rank);
        }

        // ---- (b) a rank with NO cells still joins every collective -------------------------------------------
        DeviceMesh dm;
        dm.nCells = (rank == np - 1 && np > 1) ? 0 : 4;
        dm.nInternalFaces = 0;
        dm.nBndFaces = 0;
        dm.nCellsExt = dm.nCells;
        dm.nBndFacesPhys = 0;
        dm.V = DeviceBuffer<scalar>(static_cast<std::size_t>(dm.nCells));
        dm.ownerStart = DeviceBuffer<label>(static_cast<std::size_t>(dm.nCells + 1));
        dm.losortStart = DeviceBuffer<label>(static_cast<std::size_t>(dm.nCells + 1));
        dm.losort = DeviceBuffer<label>(static_cast<std::size_t>(0));
        dm.bndCellStart = DeviceBuffer<label>(static_cast<std::size_t>(dm.nCells + 1));
        dm.bndPerm = DeviceBuffer<label>(static_cast<std::size_t>(0));
        dm.bndIsEmpty = DeviceBuffer<label>(static_cast<std::size_t>(0));
        if (dm.nCells > 0)
        {
            std::vector<scalar> ones(dm.nCells, 2.0);
            dm.V.copyFrom(ones);
            std::vector<label> start(dm.nCells + 1, 0);
            dm.ownerStart.copyFrom(start);
            dm.losortStart.copyFrom(start);
            dm.bndCellStart.copyFrom(start);
        }
        else
        {
            std::vector<label> start(1, 0);
            dm.ownerStart.copyFrom(start);
            dm.losortStart.copyFrom(start);
            dm.bndCellStart.copyFrom(start);
        }
        DeviceBuffer<scalar> phiInt(static_cast<std::size_t>(0));
        DeviceBuffer<scalar> phiBnd(static_cast<std::size_t>(0));
        const DeviceCourantNumbers dc = deviceAlphaCourantNo(dm, phiInt, phiBnd, nullptr, 0.001, nullptr, nullptr);
        check(sameOnAll(dc.CoNum, MPI_COMM_WORLD), "device Courant number differs between ranks (zero-cell rank)", rank);
        check(sameOnAll(dc.meanCoNum, MPI_COMM_WORLD), "device mean Courant differs between ranks", rank);
        const double wantCo = (np > 1) ? 0.0 : 0.0;   // no flux anywhere: every Courant number is zero
        check(dc.CoNum == wantCo && dc.meanCoNum == 0.0, "zero-flux device Courant number is not zero", rank);
        // empty reductions: global, zero contributions, no early return
        DeviceBuffer<scalar> empty(static_cast<std::size_t>(0));
        const double s = deviceSumMag(empty);
        check(s == 0.0 && sameOnAll(s, MPI_COMM_WORLD), "deviceSumMag of an empty rank is not a global zero", rank);
        const double mr = deviceMaxRatio(empty, empty);
        check(mr == 0.0 && sameOnAll(mr, MPI_COMM_WORLD), "deviceMaxRatio of an empty rank is not a global zero", rank);

        ProcComm::setCurrent(nullptr);
    }
    if (rank == 0 && fails == 0) std::printf("PASS courant_global (np=%d): identical deltaT sequences, zero-cell rank joins every reduction\n", np);
    std::fflush(stdout);
    rc = fails ? 1 : 0;
    Pstream::finalize();
    return rc;
}
