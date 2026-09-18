#pragma once
// brae::deviceGamgSolve -- OpenFOAM's GAMGSolver on the device: the V-cycle of gamg_solver_cpp.cuh
// on the hierarchy of pair_gamg_agglomeration_cpp.cuh, with the DIC smoother.
//
// provenance:
//   openfoam:  the files gamg_solver_cpp.cuh lists; this is the same transcription
//   host:      brae::gamgSolve, the in-repo oracle
//   tests:     tests/interfoam_gamg_vs_openfoam.sh, the device arm of every DIC profile
//
// WHAT RUNS WHERE.
//   THE HIERARCHY is built on the host, once, by the host's own code, and uploaded: it is mesh
//   topology, and OpenFOAM builds it once too.
//   THE COARSE MATRICES are rebuilt on the device at every solve, as GAMGSolver's constructor
//   rebuilds them, by FIXED-ORDER GATHERS: each coarse cell sums its fine cells' diagonals in
//   ascending order and then twice its interior faces' coefficients in ascending order, each coarse
//   face its fine faces ascending -- the host's accumulation order, term for term. The same for
//   restriction. A scatter with atomic adds would be neither ordered nor reproducible. (No test
//   asserts the level matrices bitwise; what holds them is the coarsest-level solve's log, which is a
//   function of the last of them -- a coarsest matrix left over from the FIRST solve of the run reads
//   2.3e-02 on that arm and moves no field.)
//   THE SMOOTHER is DIC through the level-scheduled DILU of device_dilu.cuh with lower aliased to
//   upper, which tests/test_device_dic.cu holds bit-identical to DICPreconditioner.C; one schedule per
//   level, built with the hierarchy. The other three smoothers the host runs are REFUSED here.
//   THE COARSEST LEVEL is solved on the HOST by the host's own PCG (gamgCoarsestPcgDic): ten cells
//   by default.
//   THE SCALING's two dot products and the residual norms are device reductions, so they are summed
//   in the device's order, which is the one place this differs from the host beyond the SpMV's.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_dilu.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"   // DeviceSolverPerf
#include "fv_geometry.cuh"
#include "gamg_solver_cpp.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

// one step of the hierarchy: level i -> level i+1, and level i+1's matrix and work fields
struct DeviceGamgLevel
{
    // OpenFOAM's matrixLevels_[i]; addressing uploaded once, diag and upper refilled at every solve
    DeviceLduMatrix A;
    DeviceDilu dic;
    // fine cell -> coarse cell, for the injection
    DeviceBuffer<label> restrictMap;
    // the inverse maps, CSR over the COARSE entity with the fine ones ASCENDING
    DeviceBuffer<label> cellStart;
    DeviceBuffer<label> cellList;
    DeviceBuffer<label> innerFaceStart;
    DeviceBuffer<label> innerFaceList;
    DeviceBuffer<label> faceStart;
    DeviceBuffer<label> faceList;
    // coarseCorrFields[i], coarseSources[i], and the scratch the host carves out of Apsi
    DeviceBuffer<scalar> corr;
    DeviceBuffer<scalar> source;
    DeviceBuffer<scalar> ACf;
    DeviceBuffer<scalar> preSmoothed;
    DeviceBuffer<scalar> rA;
    DeviceBuffer<scalar> wA;

    // the symmetric view: DIC is DILU with lower aliased to upper
    DeviceLduView view() const
    {
        DeviceLduView v = A.view();
        v.lower = v.upper;
        return v;
    }
};

struct DeviceGamgHierarchy
{
    // the host's hierarchy, which this mirrors; the coarsest-level solve reads its addressing
    const GamgAgglomeration* host = nullptr;
    std::vector<DeviceGamgLevel> level;
    DeviceBuffer<scalar> Apsi;
    DeviceBuffer<scalar> finestCorrection;
    DeviceBuffer<scalar> finestResidual;
    DeviceBuffer<scalar> rA;
    DeviceBuffer<scalar> wA;
};

// THE MESH'S HIERARCHY, host and device, built by the first GAMG solve of the run from THAT entry's
// nCellsInCoarsestLevel -- see GamgAgglomerationCache for why it is not the solve's.
struct DeviceGamgCache
{
    // the host mesh the hierarchy is built from; the caller sets both before the first solve
    const PrimitiveMesh* mesh = nullptr;
    const FvGeometry* geometry = nullptr;
    GamgAgglomerationCache host;
    bool uploaded = false;
    DeviceGamgHierarchy device;

    DeviceGamgHierarchy& get(label nCellsInCoarsestLevel);
};

// True for the smoothers deviceGamgSolve runs: DIC. The device driver refuses the rest by name.
bool deviceGamgSmootherPorted(const std::string& smoother);

// GAMGSolver::solve on an already FOLDED symmetric system. `fineDic` is the fine level's schedule
// (buildDeviceDilu on the mesh's own addressing); it is updated here for this matrix. Throws if the
// matrix is not symmetric or the smoother is not DIC.
DeviceSolverPerf deviceGamgSolve(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    DeviceDilu& fineDic,
    DeviceGamgHierarchy& h,
    const GamgControls& controls,
    GamgSolveLog* log = nullptr);

} // namespace brae
