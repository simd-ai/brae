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
    // the system and the solution on the device, KEPT between calls: the PCG loop's captured graph is keyed on
    // the solution's address (PCGGraphCache), so a buffer made afresh at every call would capture afresh too
    DeviceBuffer<scalar> dDiag;
    DeviceBuffer<scalar> dUpper;
    DeviceBuffer<scalar> dSource;
    DeviceBuffer<scalar> dPsi;

    // the case, for the hierarchy's disk cache (deviceAmgPcgHierarchy); empty = no disk cache
    std::string caseDir;

    // WHICH HIERARCHY pcorr's AMG-PCG runs on. pcorr starts from zero at every CorrectPhi and is solved to the
    // case's tolerance -- 1e-10 in the wave-maker tutorials -- so its cost is its iteration count, and on the
    // hierarchy p_rgh uses (pairwise aggregation, a V-cycle in single precision) that is large. MEASURED on
    // waveMakerPiston refined to 896,000 cells, 30 steps: 173 iterations a solve, 320 of CorrectPhi's 360 ms a
    // step (OpenFOAM's own DICPCG stops at its 1,000-iteration limit there, at 5e-06). On a SMOOTHED-AGGREGATION
    // hierarchy of its own: 48 iterations, 134 ms, CorrectPhi 360 -> 178, the same 30 steps to the same time.
    // The smoothed hierarchy costs ten times the plain one to build (2.1 s at 896,000 cells; 575 against 59 ms
    // a step on damBreakWithObstacle, which rebuilds at every refinement), so it is taken only where it is
    // built ONCE: `fixedTopology`, set by the caller for a mesh that moves and does not refine, and only at the
    // first build. Anywhere else pcorr takes the hierarchy p_rgh takes, as before.
    // BRAE_PCORR_AMG=plain never smooths (the gate's other arm); =sa smooths at every build, whatever the mesh.
    // NOT on the disk cache: it is rebuilt at every start.
    bool fixedTopology = false;

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
