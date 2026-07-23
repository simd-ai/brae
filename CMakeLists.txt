cmake_minimum_required(VERSION 3.24)

# Target NVIDIA GB10 / Grace Blackwell (sm_121). Override with
# -DCMAKE_CUDA_ARCHITECTURES=90 etc. on older toolkits. Phase 1 is host code in .cu
# files (compiled by nvcc); the arch matters once kernels land in the G-phases.
set(BRAE_DEFAULT_CUDA_ARCH 121)
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES ${BRAE_DEFAULT_CUDA_ARCH}
        CACHE STRING "CUDA architectures cf is built for" FORCE)
endif()

project(brae VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Per-thread default stream: makes the implicit <<<>>> stream a regular (capturable) per-thread stream, so the
# AMG V-cycle can be recorded into a CUDA graph and replayed (kills host launch overhead). cf's GPU path is
# single-threaded, so this is semantically identical to the legacy default stream. Applied to ALL TUs (no mixing).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Real MPI is the CPU-baseline communication transport (the cf::Pstream backend).
find_package(MPI REQUIRED)

# zlib, OF gzips polyMesh/fields by default (points.gz, owner.gz, ...); the mesh/field readers gunzip transparently.
find_package(ZLIB REQUIRED)

# OpenMP, parallelise embarrassingly-parallel host setup loops (the brute-force cell wall distance is
# nCells x nWallFaces; on complex geometries like motorBike the bike surface has ~40k faces -> minutes single-
# threaded). Optional: if OpenMP is absent the #pragma omp lines are ignored and the loops run serially (no-op).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
endif()

# SCOTCH, cell-graph partitioner (the decomposePar-equivalent), matching OpenFOAM.
find_path(SCOTCH_INCLUDE_DIR NAMES scotch.h PATH_SUFFIXES scotch REQUIRED)
find_library(SCOTCH_LIBRARY NAMES scotch REQUIRED)
find_library(SCOTCH_ERR_LIBRARY NAMES scotcherr scotcherrexit REQUIRED)
message(STATUS "cf SCOTCH: ${SCOTCH_LIBRARY} (hdr ${SCOTCH_INCLUDE_DIR})")

# OpenFOAM-style flat header resolution: .cuh co-located with .cu; include by basename.
# Collect every directory holding a header and add it to the include path.
file(GLOB_RECURSE BRAE_ALL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh)
set(BRAE_HEADER_DIRS "")
foreach(_h ${BRAE_ALL_HEADERS})
    get_filename_component(_d ${_h} DIRECTORY)
    list(APPEND BRAE_HEADER_DIRS ${_d})
endforeach()
list(REMOVE_DUPLICATES BRAE_HEADER_DIRS)

# ---------------------------------------------------------------------------
# brae_core, Phase 1 substrate (comm seam + interfaces). Grows as modules land.
# ---------------------------------------------------------------------------
add_library(brae_core STATIC
    src/parallel/pstream/cf_pstream.cu
    src/parallel/pstream/device_halo.cu
    src/parallel/pstream/device_reduce.cu
    src/OpenFOAM/meshes/interface/processor_interface.cu
    src/OpenFOAM/db/foam_token_reader.cu
    src/OpenFOAM/meshes/primitive_mesh.cu
    src/finiteVolume/fvMesh/fv_geometry.cu
    src/finiteVolume/fvMesh/fv_patch.cu
    src/OpenFOAM/decompose/scotch_decomposition.cu
    src/OpenFOAM/decompose/domain_decomposition.cu
    src/TurbulenceModels/wallFunctions/nut_wall_function.cu
    src/finiteVolume/finiteVolume/fvc.cu
    src/finiteVolume/finiteVolume/fvm.cu
    src/OpenFOAM/matrices/ldu_spmv.cu
    src/OpenFOAM/matrices/pcg.cu
    src/OpenFOAM/matrices/gamg.cu
    src/OpenFOAM/matrices/pbicgstab.cu
    src/cuda/device_blas.cu
    src/cuda/device_spmv.cu
    src/cuda/device_pcg.cu
    src/cuda/device_fvc.cu
    src/cuda/device_fvm.cu
    src/cuda/device_simple.cu
    src/cuda/device_boundary.cu
    src/cuda/device_divdevreff.cu
    src/cuda/device_kepsilon.cu
    src/cuda/device_komega_sst.cu
    src/finiteVolume/cfdTools/MRF/device_mrf.cu
    src/finiteVolume/cfdTools/fvOptions/device_fvoptions.cu
    src/finiteVolume/cfdTools/fvOptions/rotor_disk.cu
    src/cuda/device_amg.cu
    src/cuda/device_cluster.cu
    src/cuda/device_cyclic.cu
    src/cuda/device_ami.cu
)
target_include_directories(brae_core PUBLIC ${BRAE_HEADER_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${SCOTCH_INCLUDE_DIR})
# Link MPI libraries directly (not the imported target) so its CXX compile flags do not
# reach nvcc, which would reject them.
target_link_libraries(brae_core PUBLIC ${MPI_CXX_LIBRARIES} ${SCOTCH_LIBRARY} ${SCOTCH_ERR_LIBRARY} ZLIB::ZLIB)
# OpenMP runtime (libgomp) linked directly, like MPI above, the imported target's CXX compile flags would reach nvcc.
if(OpenMP_CXX_FOUND)
    target_link_libraries(brae_core PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# NVSHMEM device transport (multi-GPU, Phase 1+). Phase 1a is host-side only: the bootstrap
# (nvshmemx_init_attr off MPI) and the symmetric heap (nvshmem_malloc) link against libnvshmem_host and need
# NO relocatable device code. Phase 1b's device-initiated put adds -rdc + device-link on its own TU only.
# Default ON; if NVSHMEM is not found the build falls back to the MPI backend (single-GPU still compiles).
# Override the location with -DNVSHMEM_HOME=... or the NVSHMEM_HOME env var.
# ---------------------------------------------------------------------------
option(BRAE_WITH_NVSHMEM "Build the NVSHMEM device transport for multi-GPU" ON)
if(BRAE_WITH_NVSHMEM)
    find_path(NVSHMEM_INCLUDE_DIR NAMES nvshmem.h
        HINTS $ENV{NVSHMEM_HOME} ${NVSHMEM_HOME}
        PATHS /opt/nvidia/hpc_sdk/Linux_aarch64/26.5/comm_libs/13.2/nvshmem
        PATH_SUFFIXES include)
    find_library(NVSHMEM_HOST_LIB NAMES nvshmem_host
        HINTS $ENV{NVSHMEM_HOME} ${NVSHMEM_HOME}
        PATHS /opt/nvidia/hpc_sdk/Linux_aarch64/26.5/comm_libs/13.2/nvshmem
        PATH_SUFFIXES lib)
    if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_HOST_LIB)
        get_filename_component(NVSHMEM_LIB_DIR ${NVSHMEM_HOST_LIB} DIRECTORY)
        message(STATUS "cf NVSHMEM: ${NVSHMEM_HOST_LIB} (hdr ${NVSHMEM_INCLUDE_DIR})")
        target_include_directories(brae_core PUBLIC ${NVSHMEM_INCLUDE_DIR})
        target_link_libraries(brae_core PUBLIC ${NVSHMEM_HOST_LIB})
        target_compile_definitions(brae_core PUBLIC BRAE_WITH_NVSHMEM)
        # Find libnvshmem_host.so.3 and the dlopen'd MPI bootstrap plugin at runtime.
        list(APPEND CMAKE_BUILD_RPATH ${NVSHMEM_LIB_DIR})
    else()
        message(WARNING "cf NVSHMEM requested but not found; building the MPI backend only (single-GPU)")
        set(BRAE_WITH_NVSHMEM OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec REQUIRED)

add_executable(test_interface_exchange tests/test_interface_exchange.cu)
target_include_directories(test_interface_exchange PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_interface_exchange PRIVATE brae_core)

foreach(_np 1 2 4 8)
    add_test(NAME interface_exchange_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_interface_exchange>)
endforeach()

# Multi-GPU Phase 0a: rank <-> GPU binding in Pstream::init
add_executable(test_gpu_bind tests/test_gpu_bind.cu)
target_include_directories(test_gpu_bind PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_bind PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME gpu_bind_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_gpu_bind>)
endforeach()

# Multi-GPU Phase 0b: per-rank DeviceMesh upload + device-path reconstruction
add_executable(test_gpu_decompose tests/test_gpu_decompose.cu)
target_include_directories(test_gpu_decompose PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_decompose PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME gpu_decompose_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_gpu_decompose>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

# Multi-GPU Phase 1a: NVSHMEM bootstrap + symmetric heap (host-side) smoke test
if(BRAE_WITH_NVSHMEM)
    add_executable(test_nvshmem_smoke tests/test_nvshmem_smoke.cu)
    target_include_directories(test_nvshmem_smoke PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_nvshmem_smoke PRIVATE brae_core)
    foreach(_np 1 2 4)
        add_test(NAME nvshmem_smoke_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_nvshmem_smoke>)
    endforeach()

    # Phase 1b: device halo exchange (DeviceHalo/NVSHMEM) vs the host interface oracle
    add_executable(test_gpu_interface_exchange tests/test_gpu_interface_exchange.cu)
    target_include_directories(test_gpu_interface_exchange PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_gpu_interface_exchange PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_interface_exchange_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_interface_exchange>)
    endforeach()
    # NOTE: full CUDA-graph capture of the exchange is deferred to Phase 5. NVSHMEM's on-stream barrier is not
    # graph-capturable (fatal "operation not permitted when stream is capturing"); an in-graph exchange needs
    # the device-initiated put+signal path (-rdc). See docs/multi-gpu.md Phase 1b/5.

    # Phase 2: device distributed SpMV (deviceParallelAmul) vs the serial oracle
    add_executable(test_gpu_parallel_spmv tests/test_gpu_parallel_spmv.cu)
    target_include_directories(test_gpu_parallel_spmv PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_gpu_parallel_spmv PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_spmv_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_spmv>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 3: device distributed Jacobi-PCG (deviceParallelJacobiPCG) vs the single-GPU device PCG
    add_executable(test_gpu_parallel_pcg tests/test_gpu_parallel_pcg.cu)
    target_include_directories(test_gpu_parallel_pcg PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_gpu_parallel_pcg PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_pcg_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_pcg>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4a foundation: device processor-coupled fvc::gaussGrad on a decomposed mesh vs the serial gradient
    add_executable(test_gpu_parallel_grad tests/test_gpu_parallel_grad.cu)
    target_include_directories(test_gpu_parallel_grad PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_gpu_parallel_grad PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_grad_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_grad>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4a L2: device distributed Jacobi-BiCGStab (non-symmetric) vs the single-GPU device BiCGStab
    add_executable(test_gpu_parallel_bicgstab tests/test_gpu_parallel_bicgstab.cu)
    target_include_directories(test_gpu_parallel_bicgstab PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_gpu_parallel_bicgstab PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_bicgstab_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_bicgstab>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Distributed AMI mapDistribute foundation: gather reproduces the global stencil values across ranks
    add_executable(test_distributed_ami_gather tests/test_distributed_ami_gather.cu)
    target_include_directories(test_distributed_ami_gather PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_distributed_ami_gather PRIVATE brae_core)
    foreach(_np 1 2)
        add_test(NAME distributed_ami_gather_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_distributed_ami_gather>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/tc_wedge_ami)
    endforeach()

    # Phase 4a L8: device distributed momentum PREDICTOR vs the host predictor
    add_executable(test_gpu_parallel_predictor tests/test_gpu_parallel_predictor.cu)
    target_include_directories(test_gpu_parallel_predictor PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_predictor PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_predictor_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_predictor>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4a L8 driver: the device distributed SIMPLE LOOP (ParallelDeviceSimple, N iterations with a
    # maintained phi) vs the host distributed loop, + reconstructField back to the global mesh
    add_executable(test_gpu_parallel_loop tests/test_gpu_parallel_loop.cu)
    target_include_directories(test_gpu_parallel_loop PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_loop PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_loop_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_loop>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4b: the distributed kOmegaSST::correct() (2nd RAS model) reproduces np=1
    add_executable(test_gpu_parallel_komegasst tests/test_gpu_parallel_komegasst.cu)
    target_include_directories(test_gpu_parallel_komegasst PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/tests
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_komegasst PRIVATE brae_core)
    add_test(NAME gpu_parallel_komegasst_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_komegasst>
                     32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/sst_ref.txt)
    set_tests_properties(gpu_parallel_komegasst_ref PROPERTIES FIXTURES_SETUP SSTREF)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_komegasst_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_komegasst>
                         32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/sst_ref.txt)
        set_tests_properties(gpu_parallel_komegasst_np${_np} PROPERTIES FIXTURES_REQUIRED SSTREF)
    endforeach()

    # Phase 4b: the CLOSED distributed turbulent SIMPLE loop (momentum+pressure with nuEff=nu+nut, then
    # kEpsilon correct(), maintained across iterations) reproduces np=1.
    add_executable(test_gpu_parallel_turbloop tests/test_gpu_parallel_turbloop.cu)
    target_include_directories(test_gpu_parallel_turbloop PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/tests
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_turbloop PRIVATE brae_core)
    add_test(NAME gpu_parallel_turbloop_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_turbloop>
                     32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/turbloop_ref.txt)
    set_tests_properties(gpu_parallel_turbloop_ref PROPERTIES FIXTURES_SETUP TURBLOOP)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_turbloop_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_turbloop>
                         32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/turbloop_ref.txt)
        set_tests_properties(gpu_parallel_turbloop_np${_np} PROPERTIES FIXTURES_REQUIRED TURBLOOP)
    endforeach()

    # SST in the closed loop (second field = omega): same gate, sst mode
    add_test(NAME gpu_parallel_turbloop_sst_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_turbloop>
                     32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/turbloop_sst_ref.txt "" sst)
    set_tests_properties(gpu_parallel_turbloop_sst_ref PROPERTIES FIXTURES_SETUP TURBLOOPSST)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_turbloop_sst_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_turbloop>
                         32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/turbloop_sst_ref.txt "" sst)
        set_tests_properties(gpu_parallel_turbloop_sst_np${_np} PROPERTIES FIXTURES_REQUIRED TURBLOOPSST)
    endforeach()

    # Phase 4b: the full distributed kEpsilon::correct() (production + wall functions + eps/k transport + nut)
    # reproduces np=1. Duct walls are y/z, cut is x -> near-wall cells sit adjacent to the cut.
    add_executable(test_gpu_parallel_kepsilon tests/test_gpu_parallel_kepsilon.cu)
    target_include_directories(test_gpu_parallel_kepsilon PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/tests
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_kepsilon PRIVATE brae_core)
    add_test(NAME gpu_parallel_kepsilon_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_kepsilon>
                     32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/keps_ref.txt)
    set_tests_properties(gpu_parallel_kepsilon_ref PROPERTIES FIXTURES_SETUP KEPSREF)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_kepsilon_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_kepsilon>
                         32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/keps_ref.txt)
        set_tests_properties(gpu_parallel_kepsilon_np${_np} PROPERTIES FIXTURES_REQUIRED KEPSREF)
    endforeach()

    # Phase 4b foundation: the distributed scalar transport solve (deviceParallelScalarTransport) -- the
    # reusable core of the turbulence step (k/eps/omega/nuTilda) -- reproduces np=1 on a flux-carrying cut.
    add_executable(test_gpu_parallel_scalar tests/test_gpu_parallel_scalar.cu)
    target_include_directories(test_gpu_parallel_scalar PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/tests
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_scalar PRIVATE brae_core)
    add_test(NAME gpu_parallel_scalar_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_scalar>
                     32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/scalar_ref.txt)
    set_tests_properties(gpu_parallel_scalar_ref PROPERTIES FIXTURES_SETUP SCALARREF)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_scalar_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_scalar>
                         32 16 16 ${CMAKE_CURRENT_BINARY_DIR}/scalar_ref.txt)
        set_tests_properties(gpu_parallel_scalar_np${_np} PROPERTIES FIXTURES_REQUIRED SCALARREF)
    endforeach()

    # Phase 4a: the linearUpwind/LUST interface correction on a cut that CARRIES FLUX (the cavity's cut is a
    # symmetry plane -> phi~0 -> those terms are multiplied by zero and any bug in them is invisible).
    add_executable(test_gpu_parallel_duct tests/test_gpu_parallel_duct.cu)
    target_include_directories(test_gpu_parallel_duct PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/tests
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_duct PRIVATE brae_core)
    # np=1 writes the reference (no cuts), then np=2/4 must reproduce it. Sequential: np>1 needs the file.
    add_test(NAME gpu_parallel_duct_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_duct>
                     32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_ref.txt)
    set_tests_properties(gpu_parallel_duct_ref PROPERTIES FIXTURES_SETUP DUCTREF)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_duct_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_duct>
                         32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_ref.txt)
        set_tests_properties(gpu_parallel_duct_np${_np} PROPERTIES FIXTURES_REQUIRED DUCTREF)
    endforeach()

    # SHEARED duct: the same flux-carrying cut, but on a NON-ORTHOGONAL mesh (shear 0.5 -> 26.57 deg, =
    # atan(0.5)). Every other brae test mesh is orthogonal, where the cut-face geometry conventions
    # (dNei = Cf - C_remote) coincide with their normal-projected variants and cannot be told apart. This is
    # the only gate that exercises the interface terms on a non-orthogonal cut.
    add_test(NAME gpu_parallel_duct_skew_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_duct>
                     32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_skew_ref.txt nolust "" 1 0.5)
    set_tests_properties(gpu_parallel_duct_skew_ref PROPERTIES FIXTURES_SETUP DUCTSKEW)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_duct_skew_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_duct>
                         32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_skew_ref.txt nolust "" 1 0.5)
        set_tests_properties(gpu_parallel_duct_skew_np${_np} PROPERTIES FIXTURES_REQUIRED DUCTSKEW)
    endforeach()

    # NON-ORTHOGONAL correction on the sheared cut: nonOrthDeltaCoeffs implicit + the explicit corrVec.grad
    # correction (momentum) + the pEqn source AND face-flux correction. Invisible on an orthogonal mesh
    # (corrVec == 0), so this gate is the only thing standing between it and silent wrongness on real meshes.
    add_test(NAME gpu_parallel_duct_nonorth_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_duct>
                     32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_nonorth_ref.txt nolust "" 1 0.5 nonorth)
    set_tests_properties(gpu_parallel_duct_nonorth_ref PROPERTIES FIXTURES_SETUP DUCTNO)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_duct_nonorth_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_duct>
                         32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_nonorth_ref.txt nolust "" 1 0.5 nonorth)
        set_tests_properties(gpu_parallel_duct_nonorth_np${_np} PROPERTIES FIXTURES_REQUIRED DUCTNO)
    endforeach()

    # LUST (0.75 linear + 0.25 linearUpwind) over the same flux-carrying cut: it exercises BOTH interface
    # deferred corrections (linearUpwind's gradient term AND the linear term), so it is the stronger gate.
    add_test(NAME gpu_parallel_duct_lust_ref
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 1 $<TARGET_FILE:test_gpu_parallel_duct>
                     32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_lust_ref.txt lust)
    set_tests_properties(gpu_parallel_duct_lust_ref PROPERTIES FIXTURES_SETUP DUCTLUST)
    foreach(_np 2 4)
        add_test(NAME gpu_parallel_duct_lust_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np} $<TARGET_FILE:test_gpu_parallel_duct>
                         32 16 16 20 ${CMAKE_CURRENT_BINARY_DIR}/duct_lust_ref.txt lust)
        set_tests_properties(gpu_parallel_duct_lust_np${_np} PROPERTIES FIXTURES_REQUIRED DUCTLUST)
    endforeach()

    # Phase 4a EXIT GATE: the OOM proof -- a mesh too big for one GPU runs split across two. No ctest entry:
    # the mesh size is hardware-specific (it must exceed ONE GPU's VRAM), so it is driven by hand.
    add_executable(test_gpu_parallel_oom tests/test_gpu_parallel_oom.cu)
    target_include_directories(test_gpu_parallel_oom PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/tests
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_oom PRIVATE brae_core)

    # Phase 4a L8 blocker: a processor face contributes ZERO matrix coeffs on device (bcType 8 = coupled)
    add_executable(test_gpu_parallel_bc tests/test_gpu_parallel_bc.cu)
    target_include_directories(test_gpu_parallel_bc PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_gpu_parallel_bc PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_bc_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_bc>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4a L7: device processor-face pressure flux (pEqn.flux) vs host parallelMatrixFlux + conservation
    add_executable(test_gpu_parallel_pflux tests/test_gpu_parallel_pflux.cu)
    target_include_directories(test_gpu_parallel_pflux PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_pflux PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_pflux_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_pflux>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4a L5: device processor-face flux is conservative + matches host fvc::flux
    add_executable(test_gpu_parallel_flux tests/test_gpu_parallel_flux.cu)
    target_include_directories(test_gpu_parallel_flux PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(test_gpu_parallel_flux PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_flux_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_flux>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4a L4: device distributed H() vs host parallelMatrixH
    add_executable(test_gpu_parallel_h tests/test_gpu_parallel_h.cu)
    target_include_directories(test_gpu_parallel_h PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_h PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_h_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_h>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()

    # Phase 4a L1: device momentum-matrix assembly (interface coeffs) vs host momentumDistributed
    add_executable(test_gpu_parallel_momentum tests/test_gpu_parallel_momentum.cu)
    target_include_directories(test_gpu_parallel_momentum PRIVATE ${BRAE_HEADER_DIRS}
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
    target_link_libraries(test_gpu_parallel_momentum PRIVATE brae_core)
    foreach(_np 1 2 4 8)
        add_test(NAME gpu_parallel_momentum_np${_np}
                 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                         $<TARGET_FILE:test_gpu_parallel_momentum>
                         ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
    endforeach()
endif()

add_executable(test_mesh_topology tests/test_mesh_topology.cu)
target_include_directories(test_mesh_topology PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mesh_topology PRIVATE brae_core)
add_test(NAME mesh_topology
         COMMAND test_mesh_topology
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)

add_executable(test_mesh_geometry tests/test_mesh_geometry.cu)
target_include_directories(test_mesh_geometry PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mesh_geometry PRIVATE brae_core)
add_test(NAME mesh_geometry
         COMMAND test_mesh_geometry ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

add_executable(test_cell_wall_dist tests/test_cell_wall_dist.cu)
target_include_directories(test_cell_wall_dist PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cell_wall_dist PRIVATE brae_core)
add_test(NAME cell_wall_dist
         COMMAND test_cell_wall_dist ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

add_executable(test_komega_coeffs tests/test_komega_coeffs.cu)
target_include_directories(test_komega_coeffs PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_komega_coeffs PRIVATE brae_core)
add_test(NAME komega_coeffs
         COMMAND test_komega_coeffs ${CMAKE_CURRENT_SOURCE_DIR}/validation/komegaCoeffs)

add_executable(test_gpu_komega_sst tests/test_gpu_komega_sst.cu)
target_include_directories(test_gpu_komega_sst PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_komega_sst PRIVATE brae_core)
add_test(NAME gpu_komega_sst
         COMMAND test_gpu_komega_sst ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily282/komega_sst.dat)

add_executable(test_forces tests/test_forces.cu)
target_include_directories(test_forces PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_forces PRIVATE brae_core)
add_test(NAME forces
         COMMAND test_forces ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily282 282)

add_executable(test_gpu_mrf tests/test_gpu_mrf.cu)
target_include_directories(test_gpu_mrf PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_mrf PRIVATE brae_core)
add_test(NAME gpu_mrf
         COMMAND test_gpu_mrf ${CMAKE_CURRENT_SOURCE_DIR}/validation/mrfBox)


add_executable(test_decompose_exchange tests/test_decompose_exchange.cu)
target_include_directories(test_decompose_exchange PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_decompose_exchange PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME decompose_exchange_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_decompose_exchange>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_parallel_spmv tests/test_parallel_spmv.cu)
target_include_directories(test_parallel_spmv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_spmv PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_spmv_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_spmv>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_pcg tests/test_parallel_pcg.cu)
target_include_directories(test_parallel_pcg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_pcg PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_pcg_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_pcg>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_pbicgstab tests/test_parallel_pbicgstab.cu)
target_include_directories(test_parallel_pbicgstab PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_pbicgstab PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_pbicgstab_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_pbicgstab>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_local_mesh tests/test_local_mesh.cu)
target_include_directories(test_local_mesh PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_local_mesh PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME local_mesh_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_local_mesh>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_proc_delta tests/test_proc_delta.cu)
target_include_directories(test_proc_delta PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_proc_delta PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME proc_delta_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_proc_delta>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_local_assembly tests/test_local_assembly.cu)
target_include_directories(test_local_assembly PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_local_assembly PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME local_assembly_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_local_assembly>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_local_convection tests/test_local_convection.cu)
target_include_directories(test_local_convection PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_local_convection PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME local_convection_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_local_convection>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_poisson tests/test_parallel_poisson.cu)
target_include_directories(test_parallel_poisson PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_poisson PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_poisson_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_poisson>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_fields_bc tests/test_fields_bc.cu)
target_include_directories(test_fields_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fields_bc PRIVATE brae_core)
add_test(NAME fields_bc
         COMMAND test_fields_bc ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily282)

add_executable(test_wall_function tests/test_wall_function.cu)
target_include_directories(test_wall_function PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_wall_function PRIVATE brae_core)
add_test(NAME wall_function
         COMMAND test_wall_function ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily282)

add_executable(test_fvc_grad tests/test_fvc_grad.cu)
target_include_directories(test_fvc_grad PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvc_grad PRIVATE brae_core)
add_test(NAME fvc_grad
         COMMAND test_fvc_grad ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily282)

add_executable(test_fvm_laplacian tests/test_fvm_laplacian.cu)
target_include_directories(test_fvm_laplacian PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvm_laplacian PRIVATE brae_core)
add_test(NAME fvm_laplacian
         COMMAND test_fvm_laplacian ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDump)

add_executable(test_fvm_div tests/test_fvm_div.cu)
target_include_directories(test_fvm_div PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvm_div PRIVATE brae_core)
add_test(NAME fvm_div
         COMMAND test_fvm_div ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDump)

add_executable(test_spmv tests/test_spmv.cu)
target_include_directories(test_spmv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_spmv PRIVATE brae_core)
add_test(NAME spmv
         COMMAND test_spmv ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDump)

add_executable(test_pcg tests/test_pcg.cu)
target_include_directories(test_pcg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_pcg PRIVATE brae_core)
add_test(NAME pcg
         COMMAND test_pcg ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDump)

add_executable(test_gamg tests/test_gamg.cu)
target_include_directories(test_gamg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gamg PRIVATE brae_core)
add_test(NAME gamg
         COMMAND test_gamg ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpGAMG)

add_executable(test_pbicgstab tests/test_pbicgstab.cu)
target_include_directories(test_pbicgstab PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_pbicgstab PRIVATE brae_core)
add_test(NAME pbicgstab
         COMMAND test_pbicgstab ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_momentum_matrix tests/test_momentum_matrix.cu)
target_include_directories(test_momentum_matrix PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_momentum_matrix PRIVATE brae_core)
add_test(NAME momentum_matrix
         COMMAND test_momentum_matrix ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_fvc_ops tests/test_fvc_ops.cu)
target_include_directories(test_fvc_ops PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvc_ops PRIVATE brae_core)
add_test(NAME fvc_ops
         COMMAND test_fvc_ops ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_vector_solve tests/test_vector_solve.cu)
target_include_directories(test_vector_solve PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_vector_solve PRIVATE brae_core)
add_test(NAME vector_solve
         COMMAND test_vector_solve ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_peqn_matrix tests/test_peqn_matrix.cu)
target_include_directories(test_peqn_matrix PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_peqn_matrix PRIVATE brae_core)
add_test(NAME peqn_matrix
         COMMAND test_peqn_matrix ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_simple_step tests/test_simple_step.cu)
target_include_directories(test_simple_step PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_step PRIVATE brae_core)
add_test(NAME simple_step
         COMMAND test_simple_step ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpSimple)

add_executable(test_processor_field tests/test_processor_field.cu)
target_include_directories(test_processor_field PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_processor_field PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME processor_field_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_processor_field>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_simple_run tests/test_simple_run.cu)
target_include_directories(test_simple_run PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_run PRIVATE brae_core)
add_test(NAME simple_run
         COMMAND test_simple_run ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpSimple)

add_executable(test_kepsilon tests/test_kepsilon.cu)
target_include_directories(test_kepsilon PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_kepsilon PRIVATE brae_core)
add_test(NAME kepsilon
         COMMAND test_kepsilon ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily282)

add_executable(test_kepsilon_eqn tests/test_kepsilon_eqn.cu)
target_include_directories(test_kepsilon_eqn PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_kepsilon_eqn PRIVATE brae_core)
add_test(NAME kepsilon_eqn
         COMMAND test_kepsilon_eqn ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCase)

add_executable(test_kepsilon_correct tests/test_kepsilon_correct.cu)
target_include_directories(test_kepsilon_correct PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_kepsilon_correct PRIVATE brae_core)
add_test(NAME kepsilon_correct
         COMMAND test_kepsilon_correct ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)

add_executable(test_divdevreff tests/test_divdevreff.cu)
target_include_directories(test_divdevreff PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_divdevreff PRIVATE brae_core)
add_test(NAME divdevreff
         COMMAND test_divdevreff ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)

add_executable(test_simple_turbulent tests/test_simple_turbulent.cu)
target_include_directories(test_simple_turbulent PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_turbulent PRIVATE brae_core)
# Fast fidelity gates (sub-second): one cf turbulent step from OpenFOAM's converged solution must
# reproduce it to OF's convergence floor. "cap-and-measure", no thousand-iteration runs needed.
add_test(NAME turb_fixedpoint_12k
         COMMAND test_simple_turbulent ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb     1576 1576 1)
add_test(NAME turb_fixedpoint_49k
         COMMAND test_simple_turbulent ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurbBig   3000 3000 1)
# Full end-to-end convergence from 0/ (~2.5 min/12k); slow-labelled so the fast loop skips it
# (`ctest -LE slow`); run explicitly with `ctest -R simple_turbulent_full`.
add_test(NAME simple_turbulent_full
         COMMAND test_simple_turbulent ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 0 1576)
set_tests_properties(simple_turbulent_full PROPERTIES LABELS slow TIMEOUT 900)

add_executable(test_proc_interp tests/test_proc_interp.cu)
target_include_directories(test_proc_interp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_proc_interp PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME proc_interp_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_proc_interp>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_grad tests/test_parallel_grad.cu)
target_include_directories(test_parallel_grad PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_grad PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_grad_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_grad>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_transport tests/test_parallel_transport.cu)
target_include_directories(test_parallel_transport PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_transport PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_transport_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_transport>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_matrixH tests/test_parallel_matrixH.cu)
target_include_directories(test_parallel_matrixH PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_matrixH PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_matrixH_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_matrixH>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_momentum tests/test_parallel_momentum.cu)
target_include_directories(test_parallel_momentum PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_momentum PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_momentum_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_momentum>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_predictor tests/test_parallel_predictor.cu)
target_include_directories(test_parallel_predictor PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_predictor PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_predictor_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_predictor>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_simplestep tests/test_parallel_simplestep.cu)
target_include_directories(test_parallel_simplestep PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_simplestep PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_simplestep_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_simplestep>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_loop tests/test_parallel_loop.cu)
target_include_directories(test_parallel_loop PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_loop PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_loop_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_loop>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_kepsilon tests/test_parallel_kepsilon.cu)
target_include_directories(test_parallel_kepsilon PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_kepsilon PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_kepsilon_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_kepsilon>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_turbloop tests/test_parallel_turbloop.cu)
target_include_directories(test_parallel_turbloop PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_turbloop PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_turbloop_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_turbloop>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_parallel_divdevreff tests/test_parallel_divdevreff.cu)
target_include_directories(test_parallel_divdevreff PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_divdevreff PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME parallel_divdevreff_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_parallel_divdevreff>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

# Converged-run driver (not a ctest gate): mpirun -np N test_parallel_converged <case> <ofTime> <iters>
add_executable(test_parallel_converged tests/test_parallel_converged.cu)
target_include_directories(test_parallel_converged PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_parallel_converged PRIVATE brae_core)

# The production solver app: mpirun -np N brae_simpleFoam -case <caseDir>
add_executable(brae_simpleFoam src/applications/solvers/simpleFoam/simpleFoam.cu)
target_include_directories(brae_simpleFoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_simpleFoam PRIVATE brae_core)

# The single-GPU device-resident solver app: brae -case <caseDir>
add_executable(brae src/applications/solvers/simpleFoam/gpuSimpleFoam.cu)
target_include_directories(brae PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae PRIVATE brae_core)

# End-to-end kOmegaSSTLM (Langtry-Menter transition) vs a SAVED OpenFOAM v2412 oracle (no OF needed at test time):
# runs brae on a transition flat plate and checks U L2 < 3% and the gammaInt transition is captured.
# slow-labelled (~2 min); regenerate the oracle with validation/komegasstlm_wall_vs_openfoam.sh.
add_test(NAME komegasstlm_flatplate
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/validation/lm_ctest.sh
                 $<TARGET_FILE:brae> ${CMAKE_CURRENT_SOURCE_DIR}/validation/lmFlatPlate)
set_tests_properties(komegasstlm_flatplate PROPERTIES LABELS slow TIMEOUT 600)

# Distributed cyclicAMI FULL-SOLVE oracle: run tc_wedge_ami (rotational rotor-stator, linearUpwind + non-orth)
# through the single-GPU device path AND the distributed path, then compare. This is the comparison that catches
# real AMI-coupling bugs -- the bounded-div/deferred-rot/linearUpwind-grad fixes all lived in SHARED distributed
# code, so a np2-vs-np1 oracle passes with both ranks wrong; only distributed-vs-single-GPU exposes them. np=1 is
# bit-exact (tol 1e-6); np=2 exercises the cross-rank NVSHMEM AMI gather (tol 1e-3, the residualControl floor).
# _np(N)$ names auto-pick up PROCESSORS=N + (N>1) RESOURCE_LOCK gpu_multirank from the block below.
foreach(_np 1 2)
    if(_np EQUAL 1)
        set(_amitol 1e-6)
    else()
        set(_amitol 1e-3)
    endif()
    add_test(NAME ami_oracle_np${_np}
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/ami_oracle.sh
                     $<TARGET_FILE:brae>
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/tc_wedge_ami
                     ${CMAKE_CURRENT_BINARY_DIR}/ami_oracle_work_np${_np}
                     ${_np} ${_amitol} ${_amitol}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_ami_fields.py)
    set_tests_properties(ami_oracle_np${_np} PROPERTIES LABELS slow TIMEOUT 400)
endforeach()

# Cyclic increment C1: cyclic interface geometry (serial)
add_executable(test_cyclic_geometry tests/test_cyclic_geometry.cu)
target_include_directories(test_cyclic_geometry PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_geometry PRIVATE brae_core)
add_test(NAME cyclic_geometry COMMAND test_cyclic_geometry ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C2: cyclic-coupled Laplacian operator (serial)
add_executable(test_cyclic_laplacian tests/test_cyclic_laplacian.cu)
target_include_directories(test_cyclic_laplacian PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_laplacian PRIVATE brae_core)
add_test(NAME cyclic_laplacian COMMAND test_cyclic_laplacian ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C3a: CyclicFvPatchField + cyclic-aware fvc (serial)
add_executable(test_cyclic_fvc tests/test_cyclic_fvc.cu)
target_include_directories(test_cyclic_fvc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_fvc PRIVATE brae_core)
add_test(NAME cyclic_fvc COMMAND test_cyclic_fvc ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C3b: cyclic coupling inside an iterative solver (periodic Poisson)
add_executable(test_cyclic_poisson tests/test_cyclic_poisson.cu)
target_include_directories(test_cyclic_poisson PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_poisson PRIVATE brae_core)
add_test(NAME cyclic_poisson COMMAND test_cyclic_poisson ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C3c: cyclic in the full SIMPLE loop (periodic channel)
add_executable(test_cyclic_channel tests/test_cyclic_channel.cu)
target_include_directories(test_cyclic_channel PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_channel PRIVATE brae_core)
add_test(NAME cyclic_channel COMMAND test_cyclic_channel ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C-dev: the DEVICE cyclic interface operator vs the validated host (deviceAmul == cyclicAmul + symmetry)
add_executable(test_cyclic_device tests/test_cyclic_device.cu)
target_include_directories(test_cyclic_device PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_device PRIVATE brae_core)
add_test(NAME cyclic_device COMMAND test_cyclic_device ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C4: rotational transform (geometry forwardT + vector coupling)
add_executable(test_cyclic_rotational tests/test_cyclic_rotational.cu)
target_include_directories(test_cyclic_rotational PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_rotational PRIVATE brae_core)
add_test(NAME cyclic_rotational COMMAND test_cyclic_rotational ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicWedge)

# Cyclic increment C-dev-rot: the DEVICE rotational coupling (deviceCyclicFluxRot == forwardT·U[nbr], conservative)
add_executable(test_cyclic_device_rot tests/test_cyclic_device_rot.cu)
target_include_directories(test_cyclic_device_rot PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_device_rot PRIVATE brae_core)
add_test(NAME cyclic_device_rot COMMAND test_cyclic_device_rot ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicWedge)

add_executable(test_ami_weights tests/test_ami_weights.cu)
target_include_directories(test_ami_weights PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_weights PRIVATE brae_core)
add_test(NAME ami_weights COMMAND test_ami_weights)

add_executable(test_ami_device tests/test_ami_device.cu)
target_include_directories(test_ami_device PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_device PRIVATE brae_core)
add_test(NAME ami_device COMMAND test_ami_device)

add_executable(test_ami_geometry tests/test_ami_geometry.cu)
target_include_directories(test_ami_geometry PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_geometry PRIVATE brae_core)
add_test(NAME ami_geometry COMMAND test_ami_geometry ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannelAMI)

# MRF increment M1: MRFZone frame flux + Coriolis source (serial)
add_executable(test_mrf_zone tests/test_mrf_zone.cu)
target_include_directories(test_mrf_zone PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mrf_zone PRIVATE brae_core)
add_test(NAME mrf_zone COMMAND test_mrf_zone ${CMAKE_CURRENT_SOURCE_DIR}/validation/mrfBox)

# MRF increment M2+M3: MRF in the full SIMPLE loop (solid-body rotation)
add_executable(test_mrf_rotation tests/test_mrf_rotation.cu)
target_include_directories(test_mrf_rotation PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mrf_rotation PRIVATE brae_core)
add_test(NAME mrf_rotation COMMAND test_mrf_rotation ${CMAKE_CURRENT_SOURCE_DIR}/validation/mrfBox)

# GPU offload G0: device foundation (DeviceBuffer + BLAS-1 kernels)
add_executable(test_gpu_blas tests/test_gpu_blas.cu)
target_include_directories(test_gpu_blas PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_blas PRIVATE brae_core)
add_test(NAME gpu_blas COMMAND test_gpu_blas)

# GPU offload G1: device lduMatrix SpMV (Amul)
add_executable(test_gpu_spmv tests/test_gpu_spmv.cu)
target_include_directories(test_gpu_spmv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_spmv PRIVATE brae_core)
add_test(NAME gpu_spmv COMMAND test_gpu_spmv ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G2: device-resident Jacobi-PCG
add_executable(test_gpu_pcg tests/test_gpu_pcg.cu)
target_include_directories(test_gpu_pcg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_pcg PRIVATE brae_core)
add_test(NAME gpu_pcg COMMAND test_gpu_pcg ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G3: fvc explicit operators on device
add_executable(test_gpu_fvc tests/test_gpu_fvc.cu)
target_include_directories(test_gpu_fvc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_fvc PRIVATE brae_core)
add_test(NAME gpu_fvc COMMAND test_gpu_fvc ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G4: fvm matrix assembly on device
add_executable(test_gpu_fvm tests/test_gpu_fvm.cu)
target_include_directories(test_gpu_fvm PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_fvm PRIVATE brae_core)
add_test(NAME gpu_fvm COMMAND test_gpu_fvm ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G5: device pressure equation (assemble + fold + solve end-to-end)
add_executable(test_gpu_pressure tests/test_gpu_pressure.cu)
target_include_directories(test_gpu_pressure PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_pressure PRIVATE brae_core)
add_test(NAME gpu_pressure COMMAND test_gpu_pressure ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G6: device momentum predictor (assemble div-laplacian + 3-component BiCGStab solve)
add_executable(test_gpu_momentum tests/test_gpu_momentum.cu)
target_include_directories(test_gpu_momentum PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_momentum PRIVATE brae_core)
add_test(NAME gpu_momentum COMMAND test_gpu_momentum ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G7: SIMPLE coupling glue on device (H/rAU/phiHbyA/matrixFlux/corrector)
add_executable(test_gpu_simple tests/test_gpu_simple.cu)
target_include_directories(test_gpu_simple PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_simple PRIVATE brae_core)
add_test(NAME gpu_simple COMMAND test_gpu_simple ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: the closed device SIMPLE step (orchestration of G3-G7)
add_executable(test_gpu_step tests/test_gpu_step.cu)
target_include_directories(test_gpu_step PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_step PRIVATE brae_core)
add_test(NAME gpu_step COMMAND test_gpu_step ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: on-device boundary-condition evaluation
add_executable(test_gpu_boundary tests/test_gpu_boundary.cu)
target_include_directories(test_gpu_boundary PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_boundary PRIVATE brae_core)
add_test(NAME gpu_boundary COMMAND test_gpu_boundary ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: on-device vector BC evaluation (momentum)
add_executable(test_gpu_boundary_vec tests/test_gpu_boundary_vec.cu)
target_include_directories(test_gpu_boundary_vec PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_boundary_vec PRIVATE brae_core)
add_test(NAME gpu_boundary_vec COMMAND test_gpu_boundary_vec ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: boundary-flux kernels (phiHbyA.boundary, pEqn.flux().boundary)
add_executable(test_gpu_bndflux tests/test_gpu_bndflux.cu)
target_include_directories(test_gpu_bndflux PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_bndflux PRIVATE brae_core)
add_test(NAME gpu_bndflux COMMAND test_gpu_bndflux ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: the fully device-resident SIMPLE loop (residency capstone)
add_executable(test_gpu_resident tests/test_gpu_resident.cu)
target_include_directories(test_gpu_resident PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_resident PRIVATE brae_core)
add_test(NAME gpu_resident COMMAND test_gpu_resident ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: device matrix relaxation (relaxMatrix)
add_executable(test_gpu_relax tests/test_gpu_relax.cu)
target_include_directories(test_gpu_relax PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_relax PRIVATE brae_core)
add_test(NAME gpu_relax COMMAND test_gpu_relax ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload (#2): explicit divDevReff stress source on device
add_executable(test_gpu_divdevreff tests/test_gpu_divdevreff.cu)
target_include_directories(test_gpu_divdevreff PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_divdevreff PRIVATE brae_core)
add_test(NAME gpu_divdevreff COMMAND test_gpu_divdevreff ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload (#3): k-epsilon production + eddy viscosity on device
add_executable(test_gpu_kepsilon tests/test_gpu_kepsilon.cu)
target_include_directories(test_gpu_kepsilon PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_kepsilon PRIVATE brae_core)
add_test(NAME gpu_kepsilon COMMAND test_gpu_kepsilon ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload (#3): closed device kEpsilon::correct() with setValues
add_executable(test_gpu_kepsilon_correct tests/test_gpu_kepsilon_correct.cu)
target_include_directories(test_gpu_kepsilon_correct PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_kepsilon_correct PRIVATE brae_core)
add_test(NAME gpu_kepsilon_correct COMMAND test_gpu_kepsilon_correct ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: the TURBULENT device-resident driver (SIMPLE + k-epsilon, fully on GPU)
add_executable(test_gpu_resident_turb tests/test_gpu_resident_turb.cu)
target_include_directories(test_gpu_resident_turb PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_resident_turb PRIVATE brae_core)
add_test(NAME gpu_resident_turb COMMAND test_gpu_resident_turb ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload (#6): AMG preconditioner
add_executable(test_gpu_amg tests/test_gpu_amg.cu)
target_include_directories(test_gpu_amg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_amg PRIVATE brae_core)
add_test(NAME gpu_amg COMMAND test_gpu_amg ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

add_executable(test_gpu_benchmark tests/test_gpu_benchmark.cu)
target_include_directories(test_gpu_benchmark PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_benchmark PRIVATE brae_core)
add_test(NAME gpu_benchmark COMMAND test_gpu_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload #4: the device-resident driver (DeviceSimpleSolver) vs the CPU runSimple oracle, real case.
add_executable(test_gpu_driver tests/test_gpu_driver.cu)
target_include_directories(test_gpu_driver PRIVATE ${BRAE_HEADER_DIRS} src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_driver PRIVATE brae_core)
add_test(NAME gpu_driver COMMAND test_gpu_driver ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload #7: thread-block clusters + DSM, foundational cluster reduction vs the global reduction.
add_executable(test_gpu_cluster tests/test_gpu_cluster.cu)
target_include_directories(test_gpu_cluster PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_cluster PRIVATE brae_core)
add_test(NAME gpu_cluster COMMAND test_gpu_cluster)

# GPU offload #7b: AMG coarse Jacobi solve fused into one cluster+DSM kernel vs the unfused loop.
add_executable(test_gpu_amg_fused tests/test_gpu_amg_fused.cu)
target_include_directories(test_gpu_amg_fused PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_amg_fused PRIVATE brae_core)
add_test(NAME gpu_amg_fused COMMAND test_gpu_amg_fused ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload #7a: cluster+DSM SpMV vs the global-memory deviceAmul (bit-exact + timing).
add_executable(test_gpu_cluster_spmv tests/test_gpu_cluster_spmv.cu)
target_include_directories(test_gpu_cluster_spmv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_cluster_spmv PRIVATE brae_core)
add_test(NAME gpu_cluster_spmv COMMAND test_gpu_cluster_spmv ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurbBig)

# GPU offload #7c: CUDA-graph replay of the AMG V-cycle vs direct kernel launches (identical result + timing).
add_executable(test_gpu_amg_graph tests/test_gpu_amg_graph.cu)
target_include_directories(test_gpu_amg_graph PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_amg_graph PRIVATE brae_core)
add_test(NAME gpu_amg_graph COMMAND test_gpu_amg_graph ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurbBig)

# GPU offload #7c-loop: cached V-cycle graph replayed across SIMPLE steps (persistent buffers) vs direct, in-loop A/B.
add_executable(test_gpu_loop_graph tests/test_gpu_loop_graph.cu)
target_include_directories(test_gpu_loop_graph PRIVATE ${BRAE_HEADER_DIRS} src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_loop_graph PRIVATE brae_core)
add_test(NAME gpu_loop_graph COMMAND test_gpu_loop_graph ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload #8 re-measure: GPU-vs-CPU crossover with all #6/#7 wins (warm + min-of-repeats controlled timing).
add_executable(test_gpu_crossover tests/test_gpu_crossover.cu)
target_include_directories(test_gpu_crossover PRIVATE ${BRAE_HEADER_DIRS} src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_crossover PRIVATE brae_core)
add_test(NAME gpu_crossover COMMAND test_gpu_crossover ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurbBig)

# CI robustness: every MPI test is `mpirun -np N ..._npN`. Without PROCESSORS, `ctest -jK` treats each as a single
# slot and launches many concurrently -> on the self-hosted SINGLE-GPU runner that oversubscribes the GPU/CPU with
# dozens of ranks at once, the cause of the flaky failures (parallel_spmv_np2, parallel_pcg_np8 under load; each
# passes in isolation). Setting PROCESSORS=N makes the scheduler reserve N slots per test, so it never overcommits.
# PROCESSORS alone still allowed TWO np=2 tests to run at once under -j4 (4 ranks -> 4 CUDA/NVSHMEM contexts on the
# one GPU -> intermittent init/heap-contention crash, exit 8). RESOURCE_LOCK on the np>=2 tests makes them mutually
# exclusive (at most one multi-rank GPU test at a time); the many np=1 / single-GPU tests still parallelise freely.
get_property(_brae_all_tests DIRECTORY PROPERTY TESTS)
foreach(_t ${_brae_all_tests})
    if(_t MATCHES "_np([0-9]+)$")
        set_tests_properties(${_t} PROPERTIES PROCESSORS ${CMAKE_MATCH_1})
        if(CMAKE_MATCH_1 GREATER 1)
            set_tests_properties(${_t} PROPERTIES RESOURCE_LOCK gpu_multirank)
        endif()
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Demo instrumentation (demo/amgpcg): dumps the real brae AMG hierarchy and the
# per-iteration AMG-PCG state as CSV, for the replay animation. Not a test.
# ---------------------------------------------------------------------------
add_executable(trace_brae demo/amgpcg/trace_brae.cu)
target_include_directories(trace_brae PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(trace_brae PRIVATE brae_core)
