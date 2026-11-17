// ProcComm's MPI abort side channel at process teardown. Every rank destroys its communicator
// before MPI_Finalize, so an incomplete MPI_Test must leave its allgather pending until the
// same round completes; starting another round would abandon a request into the same buffer.
#include "proc_comm.cuh"
#include "cf_pstream.cuh"
#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <thread>

using namespace brae;

int main(int argc, char** argv)
{
    // SERIALIZED makes ProcComm use its main-thread pollSide() branch. The default test still
    // uses Pstream::init's THREAD_MULTIPLE and the production background-thread branch.
    int provided = MPI_THREAD_SINGLE;
    if (argc > 1 && std::string(argv[1]) == "serialized") MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo();
    const int np = Pstream::nProcs();
    if (np != 8)
    {
        std::fprintf(stderr, "test_proc_comm_teardown requires np=8, got %d\n", np);
        Pstream::finalize();
        return 1;
    }
    {
        auto comm = ProcComm::createMpiHost();
        double value = comm->allReduce(static_cast<double>(rank), ReduceOp::Sum);
        const double expected = static_cast<double>(np*(np - 1)/2);
        if (value != expected)
        {
            std::fprintf(stderr, "rank %d: reduction %.17g != %.17g\n", rank, value, expected);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        // All ranks have finished their useful work, but arrive at communicator destruction
        // at different times. No rank starts another communicator generation.
        std::this_thread::sleep_for(std::chrono::milliseconds(rank));
    }
    Pstream::finalize();
    if (rank == 0) std::printf("PASS: ProcComm process teardown at np=8\n");
    return 0;
}
