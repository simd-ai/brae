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
#include "device_fv_geometry.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"   // SurfaceScalarField
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
    // ONE BUILD A CHANGED MESH: shared with p_rgh's cache (AmgHierarchyMemo, device_inter_pressure_step.cuh),
    // which asks for the same mesh's hierarchy right after this solver does. Null = each builds its own.
    struct AmgHierarchyMemo* amgMemo = nullptr;

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
    // ...AND ONLY ON A TWO-DIMENSIONAL MESH (a patch of type empty with faces). That is what was measured, not a
    // theory: on every three-dimensional moving mesh tried the smoothed hierarchy takes as many iterations as
    // the plain one or more, at a higher price each. pcorr's iterations, smoothed / plain (2026-10-04):
    //   2-D  waveMakerPiston 896,000 cells   1,427 / 5,530 in 31 solves
    //        waveMakerPiston 56,000          99 / 196          waveMakerSolitary 14,250   79 / 107
    //   3-D  DTCHullMoving 845,536 (snappy)  626 / 354 in 10 solves; CorrectPhi 166.8 / 97.8 ms a step, and the
    //                                        smoothed one is 16 s to build or a 4.3 GB cache file
    //        DTCHullMovingCoarse 108,833     811 / 216         floatingObject 11,640 (hex)   159 / 183
    // The smoothed prolongator is shaped from a proxy of the face areas alone (device_amg.cu); why that serves a
    // structured 2-D mesh and not a snappy 3-D one is not established.
    // BRAE_PCORR_AMG=plain never smooths (the gate's other arm); =sa smooths at every build, whatever the mesh.
    // The start mesh's is read from the case's cache (deviceAmgPcgHierarchy, .brae_amgcache_sa).
    bool fixedTopology = false;

    // pcorrEqn.solve() on the GPU, whatever its entry names (`asked`, for the notice): the boundary folded as
    // fvMatrix::solveSegregated folds it, then deviceAMGPCG with the entry's stopping controls.
    // A COUPLED PAIR (cyclic, cyclicAMI) goes with it: the pair's faces, each with its own cell, its neighbour
    // slots and weights (FvPatch's own, what patchNeighbourValue sums) and the matrix's interface coefficient,
    // are put on the matrix view, where deviceAmul applies them and the AMG hierarchy carries them on every grid
    // (AMGPair). Until 2026-10-04 such a mesh was declined and pcorr solved on the host with the case's own
    // solver: MEASURED on RAS/mixerVesselAMI at 894,950 cells, CorrectPhi 364.5 ms a step.
    // Returns false, solving nothing, on a coupled patch of any other type (cyclicACMI).
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

    // THE WHOLE PASS ON THE GPU, for CorrectPhi's one pass with nNonOrthogonalCorrectors 0, where pcorr starts
    // at zero: fvm::laplacian(rAUf, pcorr) == fvc::div(phi) assembled, its boundary folded, the reference set,
    // the solve, and phi -= pcorrEqn.flux() -- the host's arithmetic operation for operation, in the host's
    // order (a cell's faces in ascending face number), so the matrix is the host's bit for bit.
    // The host built that system at every mesh update and uploaded it: MEASURED on waveMakerPiston at 896,000
    // cells, ms a step: the matrix 7.1, div(phi) 1.7, the fold and four uploads 6.0, the flux 6.6 -- 22.1 with
    // the rest, against 3.4 here (the uploads 0.7, the kernels 0.6, the flux and two downloads 0.8, and 1.2
    // comparing the mesh's owner and neighbour with the ones the addressing was built from).
    // Up: rAUf and phi on the internal faces, and for the faces of the patches that are neither empty nor
    // coupled, in patch order, the flux and the two laplacian coefficients. The geometry is the GPU's own where
    // the caller hands in one that is the host's (`deviceGeometry`, after a mesh move); uploaded otherwise.
    // Down: phi on the internal faces and pcorr's cells (for the patches' flux, which the host forms).
    // Returns false, touching nothing, on a mesh with a coupled patch: the host assembles there, and solve()
    // takes the system with its pair.
    //   `hostMatrix`  BRAE_CONTROL_PCORR_ASSEMBLY_CHECK: the system as the host assembled it; every entry of
    //                 the GPU's is compared with it, and phi with the host's flux of the GPU's solution
    bool correct(
        const std::string& asked,
        const SurfaceScalarField& rAUf,
        SurfaceScalarField& phi,
        const GeometricField<scalar>& pcorr,
        bool needReference,
        bool nonOrthDeltaCoeffs,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches,
        const DeviceFvGeometry* deviceGeometry,
        scalar tol,
        scalar relTol,
        int maxIter,
        int minIter,
        SolverPerformance& perf,
        const FvScalarMatrix* hostMatrix);

private:
    // the refusal of a coupled patch that is not a cyclic or a cyclicAMI, and the pressure rule's notice, once
    bool takes(
        const std::string& asked,
        const std::vector<FvPatch>& patches);
    // the device addressing and the hierarchy of this mesh, rebuilt where its owner or neighbour changed
    void prepare(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);

    // solve(): the coupled pair's faces up -- own cells, slots, weights, this matrix's interface coefficients
    void pairUp(
        const FvScalarMatrix& M,
        const std::vector<FvPatch>& patches);

    bool announced = false;
    bool announcedCoupled = false;
    int nPair = 0;
    DeviceBuffer<label> dPairOwn;
    DeviceBuffer<label> dPairRank;
    int nPairRanks = 1;
    DeviceBuffer<label> dPairOff;
    DeviceBuffer<label> dPairNbr;
    DeviceBuffer<scalar> dPairW;
    DeviceBuffer<scalar> dPairIfc;
    // correct(): the faces of the patches that are neither empty nor coupled, in patch order (a "slot" each),
    // and each cell's run of them
    std::size_t nSlots = 0;
    DeviceBuffer<label> dSlotStart;
    DeviceBuffer<label> dSlotList;
    DeviceBuffer<scalar> dGamma;
    DeviceBuffer<scalar> dPhi;
    DeviceBuffer<scalar> dDc;
    DeviceBuffer<scalar> dMagSf;
    DeviceBuffer<scalar> dV;
    DeviceBuffer<scalar> dSlotPhi;
    DeviceBuffer<scalar> dSlotIC;
    DeviceBuffer<scalar> dSlotBC;
};

} // namespace brae
