#pragma once
// The multi-GPU coupling's geometry stage, as the driver runs it behind the BRAE_MGPU_GEOMETRY_ONLY gate
// (inter_parallel.cuh): couple this rank's processor patches over ProcComm (one setup exchange, OpenFOAM's
// processorFvPatch formulas), build the device mesh with its halo/boundary-prefix fields and the processor
// interface, write the per-rank geometry dump tools/mgpu_oracle/compare_proc_geometry.py reads, print one summary
// line and STOP. Nothing past the geometry runs: the operators across a processor face are not ported.
//
// np == 1 needs no MPI (a one-rank loopback communicator); np > 1 is one process per rank under mpirun.
#include "case_roots.cuh"
#include "cf_pstream.cuh"
#include "device_processor.cuh"
#include "processor_coupling.cuh"
#include "processor_geometry_dump.cuh"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

inline int runProcessorGeometryOnly(
    int& argc,
    char**& argv,
    const CaseRoots& roots,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    std::vector<FvPatch>& patches)
{
    const bool mpi = roots.nProcs > 1;
    if (mpi) Pstream::init(argc, argv);
    {
        std::vector<std::unique_ptr<ProcComm>> loop;
        std::unique_ptr<ProcComm> mpiComm;
        if (mpi)
        {
            mpiComm = ProcComm::createMpiHost();
        }
        else
        {
            loop = ProcComm::createLoopback(1);
        }
        ProcComm& comm = mpi ? *mpiComm : *loop[0];
        if (comm.size() != roots.nProcs || comm.rank() != roots.rank)
        {
            throw std::runtime_error(
                "-parallel: rank " + std::to_string(roots.rank) + " of " + std::to_string(roots.nProcs)
                + " was told by the case roots, but the communicator is rank " + std::to_string(comm.rank()) + " of "
                + std::to_string(comm.size()) + ".");
        }
        ProcessorCoupling pc = attachProcessorCoupling(patches, g, m.nCells(), comm);
        const DeviceMesh dm = buildDeviceMesh(m, g, patches);
        const DeviceProcInterface dp = buildDeviceProcInterface(dm, g, patches, pc);
        const std::string dump = roots.procDir + "/brae_procGeometry.dump";
        writeProcessorGeometryDump(dump, patches, g, pc, comm);
        std::printf("brae interFoam: BRAE_MGPU_GEOMETRY_ONLY rank %d of %d: %ld cells, halo %d, %d processor faces in "
                    "%zu patch(es) with %d neighbour(s), boundary faces %d (physical %d); geometry dump %s\n",
                    comm.rank(), comm.size(), (long)m.nCells(), dp.nHalo, dp.nProcFaces, pc.patches.size(),
                    pc.plan->nNeighbours(), dm.nBndFaces, dm.nBndFacesPhys, dump.c_str());
        std::printf("brae interFoam: the operators across a processor face are not ported; stopping after the "
                    "coupling geometry.\n");
        std::fflush(stdout);
        comm.barrier();   // loopback/MPI: no rank tears its plan down before every rank is done with the dump
    }
    if (mpi) Pstream::finalize();
    return 0;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
