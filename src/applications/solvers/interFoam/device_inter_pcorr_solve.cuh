#pragma once
// THE PRESSURE RULE for pcorr (CLAUDE.md, user decisions 2026-10-03): CorrectPhi assembles pcorr's equation on
// the host on both arms (inter_correct_phi_cpp.cu); on the device arm pcorr is solved by brae's
// AMG-preconditioned PCG on the GPU, whatever its entry names -- GAMG, PCG with DIC, PCG with a GAMG
// preconditioner -- and says so.
//
// THE ADDRESSING IS ITS OWN. CorrectPhi runs inside the mesh update, and after a refinement that is BEFORE the
// device loop rebuilds its DeviceMesh (inter_driver_device.cu, interAfterMeshChange then buildDeviceMesh), so
// the loop's mesh can be the previous topology's. This solver keeps a device copy of the host mesh's
// owner/neighbour it is handed, compared by content at every call, and the AMG hierarchy built on it.
//
// NOT HERE: a mesh with a coupled patch. Its pcorr keeps the case's own solver, which carries the pair, and the
// run says so.
#include "cf_types.cuh"
#include "device_amg.cuh"
#include "device_buffer.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"   // FvScalarMatrix
#include "pcg.cuh"   // SolverPerformance
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {

struct DevicePcorrSolver
{
    // the host addressing the device copy below was built from
    std::vector<label> owner;
    std::vector<label> neighbour;
    label nCells = -1;
    unsigned long long addressingId = 0;
    DeviceBuffer<label> dOwner;
    DeviceBuffer<label> dNeighbour;
    DeviceBuffer<label> dOwnerStart;
    DeviceBuffer<label> dLosort;
    DeviceBuffer<label> dLosortStart;
    AMGData amg;

    // the case, for the hierarchy's disk cache (deviceAmgPcgHierarchy); empty = no disk cache
    std::string caseDir;

    // pcorrEqn.solve() on the GPU, whatever its entry names (`asked`, for the notice): the boundary folded as
    // fvMatrix::solveSegregated folds it, then deviceAMGPCG with the entry's stopping controls. Returns false,
    // solving nothing, on a mesh with a coupled patch.
    bool solve(
        const std::string& asked,
        const FvScalarMatrix& M,
        std::vector<scalar>& psi,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches,
        scalar tol,
        scalar relTol,
        int maxIter,
        int minIter,
        SolverPerformance& perf);
};

} // namespace brae
