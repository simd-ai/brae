#pragma once
// The multi-GPU communicator of one brae_interFoam process under `-parallel`: created from the case roots (rank and
// size come from mpirun's environment or -rank/-nprocs), installed as ProcComm::current() for the life of the object,
// torn down in the order the layers need (plans first -- they are the driver's and die with its frame -- then the
// communicator, then MPI).
//
// Backend: BRAE_COMM = mpi (default) | loopback | nccl.
//   mpi       one process per rank under `mpirun --oversubscribe -np N`: ProcComm::createMpiHost (host-staged halo,
//             deterministic gather-and-fold reductions); np = 1 needs no MPI and takes a one-rank loopback, whose
//             every operation returns on its first line.
//   loopback  np = 1 only here (the loopback harness of N rank threads in one process is the test fixtures' business)
//   nccl      ProcComm::createNccl (the communicator ProcCommNccl lands; throws by name where it is not built)
// np = 1 through this object is the same run as without -parallel: no processor patch, so nothing is coupled and no
// operation is issued.
#include "case_roots.cuh"
#include "cf_pstream.cuh"
#include "proc_comm.cuh"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

class InterProcComm
{
public:
    InterProcComm(int& argc, char**& argv, const CaseRoots& roots)
    {
        const char* e = std::getenv("BRAE_COMM");
        const std::string backend = (e != nullptr && e[0] != '\0') ? e : "mpi";
        if (backend != "mpi" && backend != "loopback" && backend != "nccl")
        {
            throw std::runtime_error("BRAE_COMM=" + backend + " is not a backend (mpi | loopback | nccl).");
        }
        if (backend == "loopback" && roots.nProcs != 1)
        {
            throw std::runtime_error(
                "BRAE_COMM=loopback runs one rank in this process, and the case is decomposed in "
                + std::to_string(roots.nProcs) + ": use BRAE_COMM=mpi under mpirun.");
        }
        mpi_ = roots.nProcs > 1;
        if (mpi_)
        {
            Pstream::init(argc, argv);
        }
        if (roots.nProcs == 1 || backend == "loopback")
        {
            owned_ = ProcComm::createLoopback(1);
        }
        else if (backend == "mpi")
        {
            owned_.push_back(ProcComm::createMpiHost());
        }
        else
        {
            owned_.push_back(ProcComm::createNccl());
        }
        comm_ = owned_[0].get();
        if (comm_->size() != roots.nProcs || comm_->rank() != roots.rank)
        {
            throw std::runtime_error(
                "-parallel: rank " + std::to_string(roots.rank) + " of " + std::to_string(roots.nProcs)
                + " was told by the case roots, but the communicator is rank " + std::to_string(comm_->rank())
                + " of " + std::to_string(comm_->size()) + ".");
        }
        ProcComm::setCurrent(comm_);
    }
    ~InterProcComm()
    {
        ProcComm::setCurrent(nullptr);
        owned_.clear();
        if (mpi_) Pstream::finalize();
    }
    InterProcComm(const InterProcComm&) = delete;
    InterProcComm& operator=(const InterProcComm&) = delete;
    ProcComm& comm() { return *comm_; }

private:
    bool mpi_ = false;
    std::vector<std::unique_ptr<ProcComm>> owned_;
    ProcComm* comm_ = nullptr;
};

}   // namespace interFoam
}   // namespace cpu
}   // namespace brae
