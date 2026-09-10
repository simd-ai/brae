cmake_minimum_required(VERSION 3.24)

# BRAE_BACKEND: CUDA (default, full solver) or ACPP (AdaptiveCpp/PCUDA, partial -- see BRAE_CUDA_ONLY_SOURCES
# below for what's excluded). Must be set before project(), since it picks which languages get enabled.
if(NOT DEFINED CACHE{BRAE_BACKEND})
    set(BRAE_BACKEND "CUDA" CACHE STRING "Compute backend: CUDA (nvcc, full solver) or ACPP (AdaptiveCpp/SYCL, portable subset)")
endif()
set_property(CACHE BRAE_BACKEND PROPERTY STRINGS CUDA ACPP)
if(NOT BRAE_BACKEND STREQUAL "CUDA" AND NOT BRAE_BACKEND STREQUAL "ACPP")
    message(FATAL_ERROR "BRAE_BACKEND must be CUDA or ACPP, got '${BRAE_BACKEND}'")
endif()

if(BRAE_BACKEND STREQUAL "CUDA")
    # Target NVIDIA GB10 / Grace Blackwell (sm_121). Override with
    # -DCMAKE_CUDA_ARCHITECTURES=90 etc. on older toolkits. Phase 1 is host code in .cu
    # files (compiled by nvcc); the arch matters once kernels land in the G-phases.
    set(BRAE_DEFAULT_CUDA_ARCH 121)
    if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES ${BRAE_DEFAULT_CUDA_ARCH}
            CACHE STRING "CUDA architectures cf is built for" FORCE)
    endif()
    project(brae VERSION 0.3.1 LANGUAGES CXX CUDA)
else()
    # acpp/syclcc is invoked as a CXX compiler (see brae_route_target_through_acpp() below), so no CUDA toolkit needed.
    project(brae VERSION 0.3.1 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BRAE_BACKEND STREQUAL "CUDA")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Per-thread default stream: makes the implicit <<<>>> stream a regular (capturable) per-thread stream, so the
    # AMG V-cycle can be recorded into a CUDA graph and replayed (kills host launch overhead). cf's GPU path is
    # single-threaded, so this is semantically identical to the legacy default stream. Applied to ALL TUs (no mixing).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
    # --extended-lambda: required for pcudaParallelFor's trampoline kernel (pcuda_compat.cuh).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
endif()

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
if(OpenMP_CXX_FOUND AND BRAE_BACKEND STREQUAL "CUDA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
endif()

if(BRAE_BACKEND STREQUAL "CUDA")
    # CUDAToolkit: the NVRTC runtime compiler (CUDA::nvrtc) + the driver API (CUDA::cuda_driver) for the
    # device-resident runtime-compiled boundary conditions (device_coded_bc.cu: nvrtc -> PTX ->
    # cuModuleLoadData -> cuLaunchKernel). No PCUDA equivalent; excluded from the ACPP build below.
    find_package(CUDAToolkit REQUIRED)
else()
    # AdaptiveCpp_DIR must point at <acpp-install>/lib/cmake/AdaptiveCpp; PCUDA needs --acpp-targets=generic.
    if(NOT DEFINED CACHE{ACPP_EXTRA_ARGS})
        set(ACPP_EXTRA_ARGS "--acpp-targets=generic --acpp-pcuda --acpp-pcuda-chevron-launch"
            CACHE STRING "Arguments to pass through directly to acpp.")
    endif()
    find_package(AdaptiveCpp CONFIG REQUIRED)
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
set(BRAE_CORE_SOURCES
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
    src/cuda/blas1.cu
    src/cuda/reductions.cu
    src/cuda/device_spmv.cu
    src/cuda/device_dilu.cu
    src/cuda/device_pcg.cu
    src/cuda/device_fvc.cu
    src/cuda/device_fvm.cu
    src/cuda/device_simple.cu
    src/cuda/device_deferred_correction.cu
    src/cuda/device_boundary_assembly.cu
    src/cuda/device_boundary_flow.cu
    src/cuda/device_divdevreff.cu
    src/cuda/device_maxwell.cu
    src/cuda/device_ddt.cu
    src/cuda/device_kepsilon.cu
    src/cuda/device_spalart.cu
    src/cuda/device_komega_sst.cu
    src/cuda/device_smagorinsky.cu
    src/cuda/device_deshybrid.cu
    src/cuda/device_generalized_newtonian.cu
    src/cuda/device_energy.cu
    src/thermophysicalModels/device_thermo.cu
    src/finiteVolume/cfdTools/MRF/device_mrf.cu
    src/finiteVolume/cfdTools/fvOptions/device_fvoptions.cu
    src/finiteVolume/cfdTools/fvOptions/rotor_disk.cu
    src/cuda/device_amg.cu
    src/cuda/device_amg_cache.cu
    src/cuda/device_amg_coarse.cu
    src/cuda/device_amg_smoothers.cu
    src/cuda/device_amg_gauss_seidel.cu
    src/cuda/device_amg_vcycle.cu
    src/cuda/device_amg_pcg.cu
    src/cuda/device_cyclic.cu
    src/cuda/device_ami.cu
    src/applications/solvers/simpleFoam/device_simple_foam.cu
)

# No PCUDA equivalent, excluded from the ACPP build:
#   device_coded_bc.cu   NVRTC + driver API (JIT-compiled user BC snippets)
#   device_cluster.cu    thread-block clusters + DSM (sm_90+ only)
set(BRAE_CUDA_ONLY_SOURCES
    src/cuda/device_coded_bc.cu
    src/cuda/device_cluster.cu
)
if(BRAE_BACKEND STREQUAL "CUDA")
    list(APPEND BRAE_CORE_SOURCES ${BRAE_CUDA_ONLY_SOURCES})
else()
    list(APPEND BRAE_CORE_SOURCES src/cuda/device_coded_bc_acpp_stub.cpp)
endif()

add_library(brae_core STATIC ${BRAE_CORE_SOURCES})
target_include_directories(brae_core PUBLIC ${BRAE_HEADER_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${SCOTCH_INCLUDE_DIR})
# Link MPI libraries directly (not the imported target) so its CXX compile flags do not
# reach nvcc, which would reject them.
target_link_libraries(brae_core PUBLIC ${MPI_CXX_LIBRARIES} ${SCOTCH_LIBRARY} ${SCOTCH_ERR_LIBRARY} ZLIB::ZLIB)
if(BRAE_BACKEND STREQUAL "CUDA")
    target_link_libraries(brae_core PUBLIC CUDA::nvrtc CUDA::cuda_driver)
endif()
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
if(BRAE_BACKEND STREQUAL "ACPP")
    set(BRAE_WITH_NVSHMEM OFF)
endif()
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
# ---------------------------------------------------------------------------------------------------
# The multi-GPU (distributed) test suite lived here, guarded by if(BRAE_WITH_NVSHMEM). It moved to
# legacy/tests/ along with the distributed solvers -- multi-GPU is OUT OF SCOPE. See legacy/README.md.
# Nothing in legacy/ is built or registered; do not re-add it without an explicit decision to resume.
# ---------------------------------------------------------------------------------------------------

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

# Transient ddt (PIMPLE foundation): deviceFvmDdt reproduces OF EulerDdtScheme/backwardDdtScheme exactly.
add_executable(test_gpu_ddt tests/test_gpu_ddt.cu)
target_include_directories(test_gpu_ddt PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ddt PRIVATE brae_core)
add_test(NAME gpu_ddt COMMAND test_gpu_ddt)

# SA-DDES length scale (DES/LES): deviceSADDESdTilda reproduces the OF SpalartAllmarasDDES delayed length scale.
add_executable(test_gpu_saddes tests/test_gpu_saddes.cu)
target_include_directories(test_gpu_saddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_saddes PRIVATE brae_core)
add_test(NAME gpu_saddes COMMAND test_gpu_saddes)

# kOmegaSST-DDES DES factor: deviceKOmegaSSTDESfactor reproduces the OF kOmegaSSTDDES k-destruction enhancement.
add_executable(test_gpu_ksstddes tests/test_gpu_ksstddes.cu)
target_include_directories(test_gpu_ksstddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ksstddes PRIVATE brae_core)
add_test(NAME gpu_ksstddes COMMAND test_gpu_ksstddes)

add_executable(test_gpu_smagorinsky tests/test_gpu_smagorinsky.cu)
target_include_directories(test_gpu_smagorinsky PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_smagorinsky PRIVATE brae_core)
add_test(NAME gpu_smagorinsky COMMAND test_gpu_smagorinsky)

add_executable(test_gpu_saiddes tests/test_gpu_saiddes.cu)
target_include_directories(test_gpu_saiddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_saiddes PRIVATE brae_core)
add_test(NAME gpu_saiddes COMMAND test_gpu_saiddes)

add_executable(test_gpu_ksstiddes tests/test_gpu_ksstiddes.cu)
target_include_directories(test_gpu_ksstiddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ksstiddes PRIVATE brae_core)
add_test(NAME gpu_ksstiddes COMMAND test_gpu_ksstiddes)

add_executable(test_gpu_ddt_cn tests/test_gpu_ddt_cn.cu)
target_include_directories(test_gpu_ddt_cn PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ddt_cn PRIVATE brae_core)
add_test(NAME gpu_ddt_cn COMMAND test_gpu_ddt_cn)

add_executable(test_gpu_coded_bc tests/test_gpu_coded_bc.cu)
target_include_directories(test_gpu_coded_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_coded_bc PRIVATE brae_core)
add_test(NAME gpu_coded_bc COMMAND test_gpu_coded_bc)

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

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
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

foreach(_np 1 2 4 8)
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

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

foreach(_np 1 2 4 8)
endforeach()

# Converged-run driver (not a ctest gate): mpirun -np N test_parallel_converged <case> <ofTime> <iters>

# brae_simpleFoam (the MPI-parallel HOST solver) moved to legacy/ -- multi-GPU is out of scope.

# The single-GPU device-resident solver app: brae -case <caseDir>
add_executable(brae src/applications/solvers/simpleFoam/gpuSimpleFoam.cu)
target_include_directories(brae PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae PRIVATE brae_core)

# The single-GPU device-resident TRANSIENT solver app (PIMPLE): brae_pimpleFoam <caseDir>. Reuses DeviceSimpleSolver's
# three composable phases via pimpleStep, with implicit fvm::ddt(U). v1 laminar.
add_executable(brae_pimpleFoam src/applications/solvers/pimpleFoam/gpuPimpleFoam.cu)
target_include_directories(brae_pimpleFoam PRIVATE ${BRAE_HEADER_DIRS} src/applications/solvers/simpleFoam)
target_link_libraries(brae_pimpleFoam PRIVATE brae_core)

# `brae` hands transient cases to brae_pimpleFoam at run time, so building brae must build it too: one target to
# build, one command to run, and no way to end up with a brae that cannot solve half the cases it is handed.
# Every solver added to the registry (solvers/common/solver_dispatch.cuh) belongs in this list.
add_dependencies(brae brae_pimpleFoam)

# End-to-end kOmegaSSTLM (Langtry-Menter transition) vs a SAVED OpenFOAM v2412 oracle (no OF needed at test time):
# runs brae on a transition flat plate and checks U L2 < 3% and the gammaInt transition is captured.
# slow-labelled (~2 min); regenerate the oracle with validation/legacy_cf/komegasstlm_wall_vs_openfoam.sh -- QUARANTINED (D5): it's still on retired cf paths and cannot run as-is.
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
endforeach()

# Single-GPU hardening regression lock: assert each fail-loud guard still fires on a mutated case
# (non-Newtonian, endTime<1, #calc, missing div(phi,U), field-size, unsupported fvOptions, wall-fn) -> a future
# edit that silently drops a guard fails here. See cudafoam/single-gpu-hardening.md.
add_test(NAME hardening_guards
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/hardening_guards.sh
                 $<TARGET_FILE:brae>
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCase
                 ${CMAKE_CURRENT_BINARY_DIR}/hardening_guards_work)

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

# brae_rhoSimpleFoam: steady compressible solver (subsonic, laminar)
add_executable(brae_rhoSimpleFoam src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu)
target_include_directories(brae_rhoSimpleFoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_rhoSimpleFoam PRIVATE brae_core)

# brae_rhoSimpleFoam_legacy: frozen snapshot of the GPU-first solver, kept runnable so the CPU-first
# ground-truth rewrite always has a reference to diff against. See rhosimplefoam-ground-truth-port.md.
# brae_rhoSimpleFoam keeps pointing at the current implementation until the new path passes every gate
# below that this one passes; only then does the name move.
add_executable(brae_rhoSimpleFoam_legacy
               src/applications/solvers/rhoSimpleFoam/legacy/gpuRhoSimpleFoamLegacy.cu)
target_include_directories(brae_rhoSimpleFoam_legacy PRIVATE ${BRAE_HEADER_DIRS}
                           src/applications/solvers/rhoSimpleFoam)
target_link_libraries(brae_rhoSimpleFoam_legacy PRIVATE brae_core)

# rhoSimpleFoam phase 4.3: alphat wall conditions load, Jayatilleke refused by name
add_executable(test_alphat_bc tests/test_alphat_bc.cu)
target_include_directories(test_alphat_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_alphat_bc PRIVATE brae_core)
add_test(NAME alphat_bc COMMAND test_alphat_bc ${CMAKE_CURRENT_SOURCE_DIR}/validation/rhoBox)

# Host boundary muEff: OF takes nut_b from the patch's own boundaryField, the legacy device path
# extrapolates the adjacent cell. Verified against nut as OpenFOAM itself wrote it -- no instrumentation.
# The test self-reports as vacuous on a case where the two candidates coincide, so the case matters.
add_executable(test_boundary_mu_eff tests/test_boundary_mu_eff.cu)
target_include_directories(test_boundary_mu_eff PRIVATE ${BRAE_HEADER_DIRS}
                           src/applications/solvers/rhoSimpleFoam/cpu)
target_link_libraries(test_boundary_mu_eff PRIVATE brae_core)
#
# THE CASE IS PART OF THE TEST. Measured over the checked-in cases, discriminating faces (where the
# patch nut and the adjacent cell nut actually differ, i.e. where the two implementations can be told
# apart) are: backwardFacingStep2D 0/41746, channel 0/20500, duct3d_cf 0/11400, airfoil 78/21812.
# rhoBox has no nut at all. So airfoil is the ONLY checked-in case that can go red here, and even it
# is weak at 0.36%. That measurement IS the explanation for how the 607x boundary-muEff defect
# survived every gate: on a blockMesh duct the extrapolated cell nut and the patch nut coincide.
# A stronger gate needs a case with a `calculated` inlet nut against a turbulent interior.
# scalarTransport gate. Until this existed NOTHING in the suite carried a tracer, and three separate
# defects lived behind that: the field solved but was never written; the convection scheme was hardcoded
# instead of read from div(phi,<field>); and a transient tracer bounded by 1.0 reached 6.32 because the
# functionObject ran before the solver knew deltaT. All three were green on the full suite throughout.
# Carries its own negative control (scheme removed under `default none` must be REFUSED), so the gate
# is proven able to fail rather than merely passing.
# Restart fidelity. A restart must reproduce a continuous run: OF pairs createPhi.H:45 READ_IF_PRESENT
# with AUTO_WRITE so phi RESUMES rather than being rebuilt from fvc::flux(U). Before the fix this case
# measured U 7.85e-06 / p 2.08e-05 against a 9.5e-12 reproducibility floor -- six orders above noise,
# shipping, and covered by nothing. Mutation-checked: reverting the phi read makes this test fail with
# exactly those numbers.
add_test(NAME restart_continuity
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/validation/restart_continuity.sh
                 $<TARGET_FILE:brae>
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/channel
                 ${CMAKE_CURRENT_BINARY_DIR}/restart_continuity_work)
set_tests_properties(restart_continuity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME tracer_transport
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/validation/tracer_transport.sh
                 $<TARGET_FILE:brae_rhoSimpleFoam>
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/tracerBox
                 ${CMAKE_CURRENT_BINARY_DIR}/tracer_transport_work)
set_tests_properties(tracer_transport PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

add_test(NAME boundary_mu_eff COMMAND test_boundary_mu_eff
         ${CMAKE_CURRENT_SOURCE_DIR}/validation/airfoil 0)
# validation/ is .gitignore'd wholesale, so the airfoil case reaches a fresh checkout only if it was
# force-added. Without this the test's 77 (its "fixture absent" code) is reported as a FAILURE, which
# reads like a solver bug rather than a missing file. Skipping is the fallback, not the intent: airfoil is
# the only checked-in case with faces where the patch nut and the cell nut differ, so a skip here means
# this gate proves nothing -- the fixture belongs in the repo.
set_tests_properties(boundary_mu_eff PROPERTIES SKIP_RETURN_CODE 77)

# rhoSimpleFoam phase 3 gate: brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam on a heated duct.
# Needs an OF install; skipped (77) if absent.
add_test(NAME rho_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/rho_vs_openfoam.sh)
set_tests_properties(rho_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam solver-control gate: SIMPLEC + nNonOrthogonalCorrectors + loose relTol, none of which
# any other compressible case sets. The compressible driver silently dropped 15 fvSolution controls until
# readLinearSolverControls; this is what makes that visible. Tolerance 2e-4 is brae's SIMPLEC floor
# (bisected: nNonOrth alone is exact at 2.6e-6), an open item recorded in the script.
add_test(NAME ctl_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/ctl_vs_openfoam.sh)
set_tests_properties(ctl_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# rhoSimpleFoam kEpsilon gate. Same duct as the kOmegaSST gate, RASModel kEpsilon.
# Tolerance is 2e-2 rather than the SST's 5e-4 for a MEASURED reason: brae's kEpsilon already differs
# from OF by k 5.09e-3 / epsilon 8.11e-3 in the INCOMPRESSIBLE solver on this same geometry
# (brae_simpleFoam vs OF simpleFoam), so that is the floor. This gate certifies the COMPRESSIBLE
# additions land on that floor, not brae's kEpsilon fidelity -- which is a separate open item.
# Dropping the rho weighting gives k 5.0e-2 / eps 2.3e-2, well clear of the floor, so it still bites.
add_test(NAME ke_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/ke_vs_openfoam.sh)
set_tests_properties(ke_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# rhoSimpleFoam BC gate: flowRateInletVelocity. OF sets U_b = avgU*n with
#   avgU = -flowRate/gSum(rho*magSf)   (flowRateInletVelocityFvPatchVectorField::updateValues)
# using the OUTWARD normal, so a positive rate is inflow. The gate runs the massFlowRate form, whose sum
# is rho-weighted and therefore recomputed every outer iteration -- the volumetric form (validation/
# rhoFRvol) is the easy half, since its sum is pure area and never changes.
#
# It also pins the bug that took the longest to find: buildDeviceVectorBoundary mapped bcCategory 9 to
# device bcType 9, an UNKNOWN type, because only the scalar buildDeviceBoundary had been taught about it.
# U is a vector field, so the inlet ran with an undefined BC and the profile bowed at the wall corners
# (row 0 5.041 vs OF 5.163) while mid-channel looked right. Reverting that one line gives T 7.7e-4.
add_test(NAME fr_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/fr_vs_openfoam.sh)
set_tests_properties(fr_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam BC gate: a totalPressure inlet on a COMPRESSIBLE case. OF's totalPressure branches on
# the dimensions of p -- kinematic gives p0 - 0.5*neg(phi)|U|^2, absolute gives p0 - 0.5*RHO*neg(phi)|U|^2.
# brae implemented only the kinematic form, so every compressible totalPressure inlet was missing the
# density on its dynamic head. No other gate uses totalPressure, so none of them could see it.
add_test(NAME tp_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/tp_vs_openfoam.sh)
set_tests_properties(tp_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam scheme gate: div(phi,h|e) and div(phi,K|Ekp) limitedLinear rather than upwind.
# brae used to HARDCODE upwind for the energy equation and ignore fvSchemes entirely -- a silent
# first-order downgrade. Every other compressible gate specifies upwind, so none of them can see it.
# Covers BOTH halves: the implicit div(phi,he) and the explicit div(phi,K|Ekp), which are separate code
# paths and were separately wrong.
add_test(NAME lu_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/lu_vs_openfoam.sh)
set_tests_properties(lu_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam transport gate: the same heated duct with transport sutherland instead of const.
# The two models compute the laminar thermal diffusivity DIFFERENTLY (const: mu/Pr; sutherland: the
# Eucken kappa/Cp, with no Pr in the dict at all), and sutherland makes mu vary strongly with T -- which
# is what exposes whether the energy laplacian uses the PATCH diffusivity or the adjacent cell's. With
# transport const those are numerically identical, so the const gates cannot see either error.
add_test(NAME suth_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/suth_vs_openfoam.sh)
set_tests_properties(suth_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam energy-form gate: the same heated duct with energy sensibleInternalEnergy (OF "e").
# Every stock OF rhoSimpleFoam tutorial uses e rather than sensibleEnthalpy, and the two forms must give
# the SAME temperature field -- OF's own h and e runs agree to 3e-11. They only do so if Cv, the gamma on
# alphaEff and the p/rho in Ekp are all right, so this gate pins three things the h gate cannot see.
add_test(NAME rhoE_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/rhoE_vs_openfoam.sh)
set_tests_properties(rhoE_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# E1 (dict_audit finding): OF controlDict write cadence on the compressible driver. Found by the audit --
# gpuRhoSimpleFoam read NONE of writeControl/writeInterval/purgeWrite/deltaT and wrote one time directory.
add_test(NAME wc_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/wc_vs_openfoam.sh)
set_tests_properties(wc_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# E5b (dict_audit, found only once the audit ran on REFUSED cases): OF RAS { turbulence off; } freezes the
# model -- k/epsilon|omega/nut keep their initial values and momentum uses that frozen nut.
add_test(NAME turbswitch_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/turbswitch_vs_openfoam.sh)
set_tests_properties(turbswitch_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# mixedEnergy: a  (Robin) T wall -- the third of OF's three energy BC families, and the only one
# that exercises OF's (1-vf) weight on refGrad.
add_test(NAME mx_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/mx_vs_openfoam.sh)
set_tests_properties(mx_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# heRhoThermo vs hePsiThermo rho TIMING at the end of pEqn.H. Compared at iteration 1, because the effect
# IS a one-iteration lag and a low-Mach case cannot distinguish the two.
add_test(NAME rhotiming_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/rhotiming_vs_openfoam.sh)
set_tests_properties(rhotiming_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# B5: heat-flux wall (fixedGradient T). The compressible gates above all use fixedValue T walls, which
# exercise only the fixedEnergy half of basicThermo::heBoundaryTypes(); a fixedGradient T was read and then
# silently discretised as zeroGradient, i.e. an adiabatic wall that converged to a plausible wrong answer.
# Also the only gate that asserts OpenFOAM can PARSE what brae writes.
add_test(NAME hf_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/hf_vs_openfoam.sh)
set_tests_properties(hf_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# rhoSimpleFoam phase 4 gate: brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam, kOmegaSST heated duct.
# Compares seven fields (T p U k omega nut rho) -- the compressible turbulence coupling is the kind that
# converges cleanly while being wrong, so residuals prove nothing here. Skipped (77) without an OF install.
# squareBend (transonic + consistent, Mach ~0.96) to CONVERGENCE vs OF. This is the evidence that
# lifted the `transonic yes` refusal: it is the only gate with an ASYMMETRIC pressure matrix (implicit
# fvm::div(phid,p) -> BiCGStab) and the only one on heRhoThermo's stored-rho timing at high Mach.
add_test(NAME transonic_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/transonic_vs_openfoam.sh)
set_tests_properties(transonic_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     LABELS slow
                     TIMEOUT 1800)

# aerofoilNACA0012 to CONVERGENCE vs OF. Slow (~700 outer iterations on both sides) and labelled so.
# It is the only gate that can catch a defect living in the pressure equation's net mass balance rather
# than in a field norm: the pinned-pressure bug reached endTime with every iteration-2 stage agreeing to
# 1e-3. See the header of the script, and cudafoam/solver-porting-dissection.md for how it was found.
add_test(NAME naca_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/naca_vs_openfoam.sh)
set_tests_properties(naca_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     LABELS slow
                     TIMEOUT 3600)

add_test(NAME sst_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/sst_vs_openfoam.sh)
set_tests_properties(sst_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# linearUpwind on the turbulence scalars -- every other gate uses upwind there, so this is the only
# one that can see which convection scheme k/omega actually run.
add_test(NAME luturb_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/luturb_vs_openfoam.sh)
set_tests_properties(luturb_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1200)

# SpalartAllmaras had NO ctest coverage; this is the first SA gate (airFoil2D, external aero + wall function).
add_test(NAME sa_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/sa_vs_openfoam.sh)
set_tests_properties(sa_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1200)

# inletOutlet on TEMPERATURE -- the only gate that exercises the energy boundary's inletOutlet mask.
add_test(NAME io_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/io_vs_openfoam.sh)
set_tests_properties(io_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# Turbulent inlet BCs on a flowRateInletVelocity inlet -- the only gate where the set-up value and the
# converged value differ, so the per-iteration refresh is actually exercised.
add_test(NAME ti_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/ti_vs_openfoam.sh)
set_tests_properties(ti_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# linearUpwind on the kinetic-energy term + grad(K)'s own boundary values (sensibleInternalEnergy).
add_test(NAME ke2_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/ke2_vs_openfoam.sh)
set_tests_properties(ke2_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# Regex-keyed boundaryField entries (alphatWallFunction Prt) -- OF patch/group/regex resolution.
add_test(NAME rx_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/rx_vs_openfoam.sh)
set_tests_properties(rx_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# pressureControl pMaxFactor/pMinFactor reference: only patches that FIX a value (OF pressureControl.C:75).
add_test(NAME pm_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/pm_vs_openfoam.sh)
set_tests_properties(pm_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 600)

# endTime is an ABSOLUTE time (OF Time::run), not a run length, and a restarted run must reproduce a
# continuous one. Both steady drivers read endTime as a length, which is only correct from startTime 0 --
# so every fresh-start gate was blind to it. Uses rhoBox: the identity needs a QUIET state, and
# angledDuct's cold start is chaotic under the pressureControl ceiling.
add_test(NAME roundtrip_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/roundtrip_vs_openfoam.sh)
set_tests_properties(roundtrip_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# flowRateInletVelocity must be fed the SOLVER's rho (the relaxed one OF registers), not thermo.rho().
# Needs a case with BOTH a relaxed rho and a flow-rate inlet; angledDuct is the only one, which is why
# this defect survived every other gate and left the case unable to converge at all.
add_test(NAME flowrate_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/flowrate_vs_openfoam.sh)
set_tests_properties(flowrate_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# RESTART fidelity: OF's createFields.H reads phi and rho with IOobject::READ_IF_PRESENT, so a restart
# resumes the STORED fields. Every other compressible gate starts from 0/, where neither file exists and
# both codes take the fallback -- which is how brae shipped without either read. This gate restarts both
# codes from the same OF-written time directory, and carries its own negative control (the same brae run
# with phi/rho deleted, which must FAIL the tolerance).
add_test(NAME restart_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/restart_vs_openfoam.sh)
set_tests_properties(restart_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 900)

# rhoSimpleFoam phase 2: mass flux reduces exactly to the volumetric flux at constant density
add_executable(test_rho_flux tests/test_rho_flux.cu)
target_include_directories(test_rho_flux PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_flux PRIVATE brae_core)
add_test(NAME rho_flux COMMAND test_rho_flux)

# rhoSimpleFoam phase 1 gate: sensible-enthalpy equation on a frozen velocity field
add_executable(test_energy_frozen tests/test_energy_frozen.cu)
target_include_directories(test_energy_frozen PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_energy_frozen PRIVATE brae_core)
add_test(NAME energy_frozen COMMAND test_energy_frozen)
# Gate 1 proper: same case, brae vs OpenFOAM scalarTransportFoam. Needs an OF install; skipped if absent.
add_test(NAME energy_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/energy_vs_openfoam.sh)
set_tests_properties(energy_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77)

# INTEGRATION: the real deviceThermoCorrect() liquid path end to end -- inversion wired to property
# evaluation. Catches the wiring errors the standalone tests structurally cannot: wrong energy
# semantics, wrong p, properties evaluated before T is updated, solver rho written instead of rhoThermo.
add_executable(test_liquid_correct tests/test_liquid_correct.cu)
target_include_directories(test_liquid_correct PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_liquid_correct PRIVATE brae_core)
add_test(NAME liquid_correct COMMAND test_liquid_correct)

# e -> T inversion (sensibleInternalEnergy), the form squareBendLiq transports. Same generic Newton,
# different F and dF/dT. 13 temperatures x 4 PRESSURES, because Es = Hs - p/rho and a single pressure
# cannot distinguish e(T) from e(p,T). Oracle is OF's thermophysicalPropertiesSelector<liquidProperties>.
add_executable(test_etot tests/test_etot.cu)
target_include_directories(test_etot PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_etot PRIVATE brae_core)
add_test(NAME etot COMMAND test_etot)

# Standalone h -> T Newton inversion for the liquid path. Targets are OF-generated enthalpies and the
# initial guesses are REVERSED (280 K target started from 400 K), so the test exercises inversion rather
# than a good guess. Accepts on the ENTHALPY residual, not |dT| -- a wrong derivative changes the path,
# not the fixed point, so |dT| alone would pass it.
add_executable(test_hetot tests/test_hetot.cu)
target_include_directories(test_hetot PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_hetot PRIVATE brae_core)
add_test(NAME hetot COMMAND test_hetot)

# Liquid thermo FIELD POPULATION: given a NONUNIFORM T field (280-400 K, one temperature per cell),
# does brae fill Cp/mu/kappa/rhoThermo per cell and per boundary face with OF's values? A uniform field
# cannot tell "evaluated per cell" from "broadcast", which is why the test refuses to use one.
add_executable(test_liquid_fields tests/test_liquid_fields.cu)
target_include_directories(test_liquid_fields PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_liquid_fields PRIVATE brae_core)
add_test(NAME liquid_fields COMMAND test_liquid_fields)

add_executable(test_liquid_fixed_bc tests/test_liquid_fixed_bc.cu)
target_include_directories(test_liquid_fixed_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_liquid_fixed_bc PRIVATE brae_core)
add_test(NAME liquid_fixed_bc COMMAND test_liquid_fixed_bc)

add_executable(test_generalized_newtonian tests/test_generalized_newtonian.cu)
target_include_directories(test_generalized_newtonian PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_generalized_newtonian PRIVATE brae_core)
add_test(NAME generalized_newtonian COMMAND test_generalized_newtonian)

# NSRDS correlation forms + H2O coefficients (squareBendLiq's `properties liquid` path) against a table
# generated by OpenFOAM ITSELF (dissect/liqref), so the test covers form and coefficients together.
# Also asserts dh/dT == Cp(T) -- the consistency a Newton he->T inversion cannot check for itself.
add_executable(test_nsrds tests/test_nsrds.cu)
target_include_directories(test_nsrds PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_nsrds PRIVATE brae_core)
add_test(NAME nsrds COMMAND test_nsrds)

# rhoSimpleFoam phase 0 gate: thermophysical subsystem (perfectGas + hConst + Sutherland/const)
add_executable(test_thermo_perfect_gas tests/test_thermo_perfect_gas.cu)
target_include_directories(test_thermo_perfect_gas PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_thermo_perfect_gas PRIVATE brae_core)
add_test(NAME thermo_perfect_gas COMMAND test_thermo_perfect_gas)

# heRhoThermo/perfectGas must parse identically to hePsiThermo, and heRhoThermo with any OTHER
# equationOfState must be refused. Asserted at the PARSE level because brae is not run-to-run
# bit-reproducible (~3e-9 on T after one iteration), so a solver-level equivalence check would need a
# statistical argument; the claim itself is deterministic.
add_executable(test_thermo_type_equiv tests/test_thermo_type_equiv.cu)
target_include_directories(test_thermo_type_equiv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_thermo_type_equiv PRIVATE brae_core)
add_test(NAME thermo_type_equiv COMMAND test_thermo_type_equiv)

add_executable(test_linear_solver_setup tests/test_linear_solver_setup.cu)
target_include_directories(test_linear_solver_setup PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_linear_solver_setup PRIVATE brae_core)
add_test(NAME linear_solver_setup COMMAND test_linear_solver_setup)

# D2: boundary coefficients (internalCoeffs/boundaryCoeffs) vs OF across four REAL BC types at once.
add_test(NAME bc_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/bc_vs_openfoam.sh)
set_tests_properties(bc_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS bcoeff_compare
                     TIMEOUT 600)

# D5: every validation/*_vs_openfoam.sh must be registered, runnable and asserting. 27 were none of those.
add_executable(test_gate_registration tests/test_gate_registration.cu)
target_include_directories(test_gate_registration PRIVATE ${BRAE_HEADER_DIRS})
add_test(NAME gate_registration COMMAND test_gate_registration ${CMAKE_CURRENT_SOURCE_DIR})

# E6: the universal gas constant must be RESOLVED like OF (DimensionedConstants NA*k), not compiled in.
add_executable(test_sensible_energy tests/test_sensible_energy.cu)
target_include_directories(test_sensible_energy PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_sensible_energy PRIVATE brae_core)
add_test(NAME sensible_energy COMMAND test_sensible_energy)

add_executable(test_foam_constants tests/test_foam_constants.cu)
target_include_directories(test_foam_constants PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_foam_constants PRIVATE brae_core)
add_test(NAME foam_constants COMMAND test_foam_constants)

# E2/E3 (dict_audit): brae must SAY when it substitutes its own linear solver / preconditioner.
add_executable(test_solver_notices tests/test_solver_notices.cu)
target_include_directories(test_solver_notices PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_solver_notices PRIVATE brae_core)
add_test(NAME solver_notices COMMAND test_solver_notices)

# Group C: fvSchemes entries must resolve within their own sub-dictionary (C1-C4).
add_executable(test_scheme_blocks tests/test_scheme_blocks.cu)
target_include_directories(test_scheme_blocks PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_scheme_blocks PRIVATE brae_core)
add_test(NAME scheme_blocks COMMAND test_scheme_blocks)

add_executable(test_residual_control tests/test_residual_control.cu)
target_include_directories(test_residual_control PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_residual_control PRIVATE brae_core)
add_test(NAME residual_control COMMAND test_residual_control)

add_executable(test_overset_refused tests/test_overset_refused.cu)
target_include_directories(test_overset_refused PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_overset_refused PRIVATE brae_core)
add_executable(test_uniform_function1 tests/test_uniform_function1.cu)
target_include_directories(test_uniform_function1 PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_uniform_function1 PRIVATE brae_core)
add_test(NAME uniform_function1 COMMAND test_uniform_function1)

add_test(NAME overset_refused
         COMMAND test_overset_refused ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# diag_compare: DIAGNOSTIC, not a test. Assembles the same scalar equation as the OpenFOAM utility
# in tools/dumpScalarMatrix and compares the matrix cell by cell. Needs OF's Ddiag/Dsource in the case,
# so it is built but never registered with ctest.
# D2: boundary coefficients (internalCoeffs/boundaryCoeffs) vs OF for REAL BC types. D1 compared only the
# diagonal and source, and only with fixedValue/zeroGradient.
# Phase 0 stage harness: compare one SIMPLE stage brae-vs-OF in dependency order.
add_executable(stage_compare tests/stage_compare.cu)
target_include_directories(stage_compare PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(stage_compare PRIVATE brae_core)

add_executable(bcoeff_compare tests/bcoeff_compare.cu)
target_include_directories(bcoeff_compare PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(bcoeff_compare PRIVATE brae_core)

add_executable(diag_compare tests/diag_compare.cu)
target_include_directories(diag_compare PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(diag_compare PRIVATE brae_core)

add_executable(test_linear_upwind_const tests/test_linear_upwind_const.cu)
target_include_directories(test_linear_upwind_const PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_linear_upwind_const PRIVATE brae_core)
add_test(NAME linear_upwind_const
         COMMAND test_linear_upwind_const ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)

# Cyclic increment C3b: cyclic coupling inside an iterative solver (periodic Poisson)
add_executable(test_cyclic_poisson tests/test_cyclic_poisson.cu)
target_include_directories(test_cyclic_poisson PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cyclic_poisson PRIVATE brae_core)
add_test(NAME cyclic_poisson COMMAND test_cyclic_poisson ${CMAKE_CURRENT_SOURCE_DIR}/validation/cyclicChannel)

# Cyclic increment C3c: cyclic in the full SIMPLE loop (periodic channel)
# phi's coupled-patch flux must survive write -> read, or an AMI/cyclic restart silently rebuilds the
# interface flux from U and lands the difference on the momentum diagonal (see the test's header).
# meshPhi / fvc::makeRelative: a fluid moving WITH the mesh must have zero relative flux, on the
# cyclicACMI interface faces as much as anywhere else (see the test's header).
# fvc::ddtCorr: the ddt flux correction pimpleFoam adds to phiHbyA (see the test's header).
# fvc::correctUf: Sf & Uf == phi, and why phi + mesh.phi() cannot stand in for it (see the header).
# A cyclicACMI non-overlap patch is a wall only where it is closed (see the test's header).
add_executable(test_acmi_wall_gate tests/test_acmi_wall_gate.cu)
target_include_directories(test_acmi_wall_gate PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_acmi_wall_gate PRIVATE brae_core)
add_test(NAME acmi_wall_gate COMMAND test_acmi_wall_gate)
set_tests_properties(acmi_wall_gate PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_wall_data_refresh tests/test_wall_data_refresh.cu)
target_include_directories(test_wall_data_refresh PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_wall_data_refresh PRIVATE brae_core)
add_test(NAME wall_data_refresh COMMAND test_wall_data_refresh)
set_tests_properties(wall_data_refresh PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_cell_limit_interface tests/test_cell_limit_interface.cu)
target_include_directories(test_cell_limit_interface PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_cell_limit_interface PRIVATE brae_core)
add_test(NAME cell_limit_interface COMMAND test_cell_limit_interface)
set_tests_properties(cell_limit_interface PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_scalar_interface_corr tests/test_scalar_interface_corr.cu)
target_include_directories(test_scalar_interface_corr PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_scalar_interface_corr PRIVATE brae_core)
add_test(NAME scalar_interface_corr COMMAND test_scalar_interface_corr)
set_tests_properties(scalar_interface_corr PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_kepsilon_production_and_blend tests/test_kepsilon_production_and_blend.cu)
target_include_directories(test_kepsilon_production_and_blend PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_kepsilon_production_and_blend PRIVATE brae_core)
add_test(NAME kepsilon_production_and_blend COMMAND test_kepsilon_production_and_blend)

add_executable(test_near_wall_dist_combined tests/test_near_wall_dist_combined.cu)
target_include_directories(test_near_wall_dist_combined PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_near_wall_dist_combined PRIVATE brae_core)
add_test(NAME near_wall_dist_combined COMMAND test_near_wall_dist_combined)

add_executable(test_boundary_subdict_skip tests/test_boundary_subdict_skip.cu)
target_include_directories(test_boundary_subdict_skip PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_boundary_subdict_skip PRIVATE brae_core)
add_test(NAME boundary_subdict_skip COMMAND test_boundary_subdict_skip)

add_executable(test_fvoptions_sources_key tests/test_fvoptions_sources_key.cu)
target_include_directories(test_fvoptions_sources_key PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_fvoptions_sources_key PRIVATE brae_core)
add_test(NAME fvoptions_sources_key COMMAND test_fvoptions_sources_key)


add_executable(test_wedge_patch tests/test_wedge_patch.cu)
target_include_directories(test_wedge_patch PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_wedge_patch PRIVATE brae_core)
add_test(NAME wedge_patch COMMAND test_wedge_patch)

add_executable(test_ami_broad_phase tests/test_ami_broad_phase.cu)
target_include_directories(test_ami_broad_phase PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_ami_broad_phase PRIVATE brae_core)
add_test(NAME ami_broad_phase COMMAND test_ami_broad_phase)

add_executable(test_courant_device tests/test_courant_device.cu)
target_include_directories(test_courant_device PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_courant_device PRIVATE brae_core)
add_test(NAME courant_device COMMAND test_courant_device)

add_executable(test_lust_implicit tests/test_lust_implicit.cu)
target_include_directories(test_lust_implicit PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_lust_implicit PRIVATE brae_core)
add_test(NAME lust_implicit COMMAND test_lust_implicit)
add_executable(test_dilu tests/test_dilu.cu)
target_include_directories(test_dilu PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_dilu PRIVATE brae_core)
add_test(NAME dilu COMMAND test_dilu)
add_executable(test_relaxation_final tests/test_relaxation_final.cu)
target_include_directories(test_relaxation_final PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_relaxation_final PRIVATE brae_core)
add_test(NAME relaxation_final COMMAND test_relaxation_final)
add_executable(test_mean_velocity_force tests/test_mean_velocity_force.cu)
target_include_directories(test_mean_velocity_force PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_mean_velocity_force PRIVATE brae_core)
add_test(NAME mean_velocity_force COMMAND test_mean_velocity_force)
add_executable(test_interface_invariants_real tests/test_interface_invariants_real.cu)
target_include_directories(test_interface_invariants_real PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_interface_invariants_real PRIVATE brae_core)
add_test(NAME interface_invariants_real COMMAND test_interface_invariants_real)

add_executable(test_interface_invariants tests/test_interface_invariants.cu)
target_include_directories(test_interface_invariants PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_interface_invariants PRIVATE brae_core)
add_test(NAME interface_invariants COMMAND test_interface_invariants)

add_executable(test_motion_contract tests/test_motion_contract.cu)
target_include_directories(test_motion_contract PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_motion_contract PRIVATE brae_core)
add_test(NAME motion_contract COMMAND test_motion_contract)

add_executable(test_ddt_corr_backward tests/test_ddt_corr_backward.cu)
target_include_directories(test_ddt_corr_backward PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_ddt_corr_backward PRIVATE brae_core)
add_test(NAME ddt_corr_backward COMMAND test_ddt_corr_backward)

add_executable(test_pimple_loop_contract tests/test_pimple_loop_contract.cu)
target_include_directories(test_pimple_loop_contract PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_pimple_loop_contract PRIVATE brae_core)
add_test(NAME pimple_loop_contract COMMAND test_pimple_loop_contract)

add_executable(test_periodic_ami tests/test_periodic_ami.cu)
target_include_directories(test_periodic_ami PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_periodic_ami PRIVATE brae_core)
add_test(NAME periodic_ami COMMAND test_periodic_ami)

add_executable(test_coupled_patch_refusal tests/test_coupled_patch_refusal.cu)
target_include_directories(test_coupled_patch_refusal PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_coupled_patch_refusal PRIVATE brae_core)
add_test(NAME coupled_patch_refusal COMMAND test_coupled_patch_refusal)

add_executable(test_correct_phi tests/test_correct_phi.cu)
target_include_directories(test_correct_phi PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_correct_phi PRIVATE brae_core)
add_test(NAME correct_phi COMMAND test_correct_phi)

add_executable(test_rng_kepsilon tests/test_rng_kepsilon.cu)
target_include_directories(test_rng_kepsilon PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_rng_kepsilon PRIVATE brae_core)
add_test(NAME rng_kepsilon COMMAND test_rng_kepsilon)

add_executable(test_wedge_device_coeffs tests/test_wedge_device_coeffs.cu)
target_include_directories(test_wedge_device_coeffs PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_wedge_device_coeffs PRIVATE brae_core)
add_test(NAME wedge_device_coeffs COMMAND test_wedge_device_coeffs)

add_executable(test_vol_point_interp tests/test_vol_point_interp.cu)
target_include_directories(test_vol_point_interp PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_vol_point_interp PRIVATE brae_core)
add_test(NAME vol_point_interp COMMAND test_vol_point_interp)

add_executable(test_eval_scoped tests/test_eval_scoped.cu)
target_include_directories(test_eval_scoped PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_eval_scoped PRIVATE brae_core)
add_test(NAME eval_scoped COMMAND test_eval_scoped)

add_executable(test_maxwell_terms tests/test_maxwell_terms.cu)
target_include_directories(test_maxwell_terms PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_maxwell_terms PRIVATE brae_core)
add_test(NAME maxwell_terms COMMAND test_maxwell_terms)

add_executable(test_acmi_scale tests/test_acmi_scale.cu)
target_include_directories(test_acmi_scale PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_acmi_scale PRIVATE brae_core)
add_test(NAME acmi_scale COMMAND test_acmi_scale)
set_tests_properties(acmi_scale PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_zdes_shielding tests/test_zdes_shielding.cu)
target_include_directories(test_zdes_shielding PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_zdes_shielding PRIVATE brae_core)
add_test(NAME zdes_shielding COMMAND test_zdes_shielding)

add_executable(test_hbya_inletoutlet tests/test_hbya_inletoutlet.cu)
target_include_directories(test_hbya_inletoutlet PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_hbya_inletoutlet PRIVATE brae_core)
add_test(NAME hbya_inletoutlet COMMAND test_hbya_inletoutlet)

add_executable(test_limited_linear_v tests/test_limited_linear_v.cu)
target_include_directories(test_limited_linear_v PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_limited_linear_v PRIVATE brae_core)
add_test(NAME limited_linear_v COMMAND test_limited_linear_v)

add_executable(test_max_delta_xyz tests/test_max_delta_xyz.cu)
target_include_directories(test_max_delta_xyz PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_max_delta_xyz PRIVATE brae_core)
add_test(NAME max_delta_xyz COMMAND test_max_delta_xyz)

add_executable(test_wale_nut tests/test_wale_nut.cu)
target_include_directories(test_wale_nut PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_wale_nut PRIVATE brae_core)
add_test(NAME wale_nut COMMAND test_wale_nut)

add_executable(test_gz_field_write tests/test_gz_field_write.cu)
target_include_directories(test_gz_field_write PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_gz_field_write PRIVATE brae_core)
add_test(NAME gz_field_write COMMAND test_gz_field_write)

add_executable(test_exact_wall_dist tests/test_exact_wall_dist.cu)
target_include_directories(test_exact_wall_dist PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_exact_wall_dist PRIVATE brae_core)
add_test(NAME exact_wall_dist COMMAND test_exact_wall_dist)

add_executable(test_fan_pressure tests/test_fan_pressure.cu)
target_include_directories(test_fan_pressure PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_fan_pressure PRIVATE brae_core)
add_test(NAME fan_pressure COMMAND test_fan_pressure)

add_executable(test_deshybrid tests/test_deshybrid.cu)
target_include_directories(test_deshybrid PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_deshybrid PRIVATE brae_core)
add_test(NAME deshybrid COMMAND test_deshybrid)

add_executable(test_binary_field_read tests/test_binary_field_read.cu)
target_include_directories(test_binary_field_read PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_binary_field_read PRIVATE brae_core)
add_test(NAME binary_field_read COMMAND test_binary_field_read)

add_executable(test_fixed_mean tests/test_fixed_mean.cu)
target_include_directories(test_fixed_mean PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_fixed_mean PRIVATE brae_core)
add_test(NAME fixed_mean COMMAND test_fixed_mean)

# divDevReff must use the case's NAMED grad(U) scheme, not a bare Gauss one (see the test's header).
add_executable(test_divdevreff_gradscheme tests/test_divdevreff_gradscheme.cu)
target_include_directories(test_divdevreff_gradscheme PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_divdevreff_gradscheme PRIVATE brae_core)
add_test(NAME divdevreff_gradscheme COMMAND test_divdevreff_gradscheme)

add_executable(test_correct_uf tests/test_correct_uf.cu)
target_include_directories(test_correct_uf PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_correct_uf PRIVATE brae_core)
add_test(NAME correct_uf COMMAND test_correct_uf)

add_executable(test_ddt_corr tests/test_ddt_corr.cu)
target_include_directories(test_ddt_corr PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ddt_corr PRIVATE brae_core)
add_test(NAME ddt_corr COMMAND test_ddt_corr)

add_executable(test_mesh_flux_relative tests/test_mesh_flux_relative.cu)
target_include_directories(test_mesh_flux_relative PRIVATE ${BRAE_HEADER_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_mesh_flux_relative PRIVATE brae_core)
add_test(NAME mesh_flux_relative COMMAND test_mesh_flux_relative)
set_tests_properties(mesh_flux_relative PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_coupled_phi_roundtrip tests/test_coupled_phi_roundtrip.cu)
target_include_directories(test_coupled_phi_roundtrip PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_coupled_phi_roundtrip PRIVATE brae_core)
add_test(NAME coupled_phi_roundtrip COMMAND test_coupled_phi_roundtrip ${CMAKE_CURRENT_BINARY_DIR}/coupled_phi_roundtrip_work)

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

add_executable(test_acmi_area_scaling tests/test_acmi_area_scaling.cu)
target_include_directories(test_acmi_area_scaling PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_acmi_area_scaling PRIVATE brae_core)
add_test(NAME acmi_area_scaling COMMAND test_acmi_area_scaling)
set_tests_properties(acmi_area_scaling PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_acmi_mask tests/test_acmi_mask.cu)
target_include_directories(test_acmi_mask PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_acmi_mask PRIVATE brae_core)
add_test(NAME acmi_mask COMMAND test_acmi_mask)
set_tests_properties(acmi_mask PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_acmi_patch_read tests/test_acmi_patch_read.cu)
target_include_directories(test_acmi_patch_read PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_acmi_patch_read PRIVATE brae_core)
add_test(NAME acmi_patch_read COMMAND test_acmi_patch_read)

add_executable(test_ami_curved tests/test_ami_curved.cu)
target_include_directories(test_ami_curved PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ami_curved PRIVATE brae_core)
add_test(NAME ami_curved COMMAND test_ami_curved)

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

# PIMPLE foundation: transient pimpleStep() steady-limit consistency vs step() + transient stability (skips if the
# case fixture is absent, like ami_oracle -- validation/ is local-only on CI).
add_executable(test_gpu_pimple tests/test_gpu_pimple.cu)
target_include_directories(test_gpu_pimple PRIVATE ${BRAE_HEADER_DIRS} src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_pimple PRIVATE brae_core)
add_test(NAME gpu_pimple COMMAND test_gpu_pimple ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(gpu_pimple PROPERTIES SKIP_RETURN_CODE 125)

# Driver-level ctest: run the brae_pimpleFoam EXECUTABLE end-to-end on a transient variant of the committed pitzDaily
# (kEpsilon URANS) -- exercises the controlDict/fvSchemes(ddt)/fvSolution(PIMPLE) parse + turbulence I/O + time loop +
# OF writer. Fixture is committed, so this runs on CI (skips only if it is somehow absent).
add_test(NAME pimple_run
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/pimple_run.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily
                 ${CMAKE_CURRENT_BINARY_DIR}/pimple_run_work)
set_tests_properties(pimple_run PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 300)

# PIMPLE/momentumPredictor: the same case with the switch on and off must give DIFFERENT fields (and the
# default must equal `on`). See the header of the script for why the log line is not enough.
add_test(NAME momentum_predictor
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/momentum_predictor.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily
                 ${CMAKE_CURRENT_BINARY_DIR}/momentum_predictor_work)
set_tests_properties(momentum_predictor PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 300)

# The same transient case run through `brae` itself: the case says `application pimpleFoam`, so the solver
# registry (solvers/common/solver_dispatch.cuh) must hand it to brae_pimpleFoam and produce the identical
# end-to-end result. Guards the single-command CLI: `brae` on a transient case must not fall through to SIMPLE.
add_test(NAME pimple_dispatch
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/pimple_run.sh
                 $<TARGET_FILE:brae>
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily
                 ${CMAKE_CURRENT_BINARY_DIR}/pimple_dispatch_work)
set_tests_properties(pimple_dispatch PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 300)

# The transient solver's refusals: MRF / fvOptions / adjustTimeStep are applied by the STEADY driver and not yet
# by this one, so a case using them must stop rather than be marched with the physics quietly missing. Includes
# the negative controls (an INACTIVE MRF zone, and a clean case) so the guards cannot creep. Dict-only, no GPU.
add_test(NAME pimple_guards
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/pimple_guards.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${CMAKE_CURRENT_BINARY_DIR}/pimple_guards_work)
set_tests_properties(pimple_guards PROPERTIES TIMEOUT 120)

# The CLI's two reserved words (`benchmark`, `node`) route to their subcommands, and -- the half that protects
# every existing user -- nothing else does: a case directory called `run`, `status` or `list` is still a case.
add_test(NAME brae_subcommands
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/brae_subcommands.sh
                 $<TARGET_FILE:brae>
                 ${CMAKE_CURRENT_BINARY_DIR}/brae_subcommands_work)
set_tests_properties(brae_subcommands PROPERTIES TIMEOUT 300)

# `brae benchmark <sample>` end to end against a LOCAL fixture template repo: fetch, cache, manifest, and the
# security guard that refuses a sample carrying coded (NVRTC-compiled) boundary conditions. Solves the committed
# pitzDaily cut to 5 iterations, so the happy path is a real run.
add_test(NAME brae_benchmark
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/brae_benchmark.sh
                 $<TARGET_FILE:brae>
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily
                 ${CMAKE_CURRENT_BINARY_DIR}/brae_benchmark_work)
set_tests_properties(brae_benchmark PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 600)

# brae-agent unit tests (milestone 3): JSON, SHA-256, the identity file, sample-name validation, argv
# construction and backoff. Deliberately a CXX-ONLY target that links neither brae_core nor CUDA -- the agent
# must build and run on a machine whose solver build is broken, and so must its tests. No GPU, no network.
add_executable(test_node_agent tests/test_node_agent.cpp
               src/node/json.cpp src/node/jobs.cpp src/node/sha256.cpp src/node/identity.cpp)
target_include_directories(test_node_agent PRIVATE src/node)
set_target_properties(test_node_agent PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_agent COMMAND test_node_agent)
set_tests_properties(node_agent PROPERTIES TIMEOUT 60)

# The agent's protocol against a FAKE transport: every status code, a body that is not JSON, JSON of the wrong
# shape, an assignment for a job type this agent does not know. Those are the cases a real orchestrator will not
# produce on request, and they decide whether a node quietly stops working. No network.
add_executable(test_node_api tests/test_node_api.cpp
               src/node/api_client.cpp src/node/json.cpp src/node/jobs.cpp)
target_include_directories(test_node_api PRIVATE src/node)
set_target_properties(test_node_api PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_api COMMAND test_node_api)
set_tests_properties(node_api PROPERTIES TIMEOUT 60)

# Job execution and GPU detection. The runner is driven against fake `brae` binaries the test writes itself --
# one that succeeds, one that fails, one that hangs, one that is missing -- because those four are what actually
# happen on a contributor's machine. The GPU probe covers the no-driver path and, where there is a driver, the
# real one. Needs dl for NVML; still no CUDA and no network.
add_executable(test_node_runner tests/test_node_runner.cpp
               src/node/runner.cpp src/node/gpu_probe.cpp src/node/jobs.cpp src/node/sha256.cpp)
target_include_directories(test_node_runner PRIVATE src/node)
target_link_libraries(test_node_runner PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(test_node_runner PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_runner COMMAND test_node_runner)
set_tests_properties(node_runner PROPERTIES TIMEOUT 120)

# ---------------------------------------------------------------------------------------------------------
# brae-agent: the node service, and the implementation behind `brae node ...`.
#
# A CXX-ONLY target. It links neither CUDA nor brae_core, deliberately: a contributor's machine must keep
# reporting even when the solver build on it is broken, and a solver bug must not be able to take the node's
# reporting down with it. libcurl and dlopen are its entire dependency budget.
# ---------------------------------------------------------------------------------------------------------
set(BRAE_AGENT_SOURCES
    src/node/agent_main.cpp src/node/agent.cpp src/node/cli.cpp src/node/service_systemd.cpp
    src/node/api_client.cpp src/node/http_curl.cpp src/node/json.cpp src/node/jobs.cpp
    src/node/identity.cpp src/node/gpu_probe.cpp src/node/runner.cpp src/node/sha256.cpp)

find_package(Threads REQUIRED)          # the executor runs a job on a worker thread so heartbeats continue
find_package(CURL QUIET)
if(CURL_FOUND)
    add_executable(brae-agent ${BRAE_AGENT_SOURCES})
    target_include_directories(brae-agent PRIVATE src/node)
    target_link_libraries(brae-agent PRIVATE CURL::libcurl Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(brae-agent PROPERTIES LINKER_LANGUAGE CXX)
    # `brae node <verb>` execs brae-agent from beside its own binary, so the two must land together.
    add_dependencies(brae brae-agent)
else()
    message(STATUS "brae: libcurl not found -- brae-agent is not built "
                   "(install libcurl4-openssl-dev); the solver is unaffected")
endif()

# The loop itself, driven at exact instants with a fake transport, fake executor and fake clock: a job that
# outlives several heartbeats, a result POST that fails repeatedly, a process restarted between finishing a job
# and reporting it, a 401 after months of working. No threads, no sleeps, no network.
add_executable(test_node_loop tests/test_node_loop.cpp
               src/node/agent.cpp src/node/api_client.cpp src/node/json.cpp
               src/node/jobs.cpp src/node/runner.cpp)
target_include_directories(test_node_loop PRIVATE src/node)
target_link_libraries(test_node_loop PRIVATE Threads::Threads)
set_target_properties(test_node_loop PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_loop COMMAND test_node_loop)
set_tests_properties(node_loop PROPERTIES TIMEOUT 120)

# `brae node register | status | unregister`, against a fake API and a fake service manager. The criterion is
# atomicity: each step of registration is made to fail in turn -- the GPU probe, the API, the identity write,
# the service account, the ownership hand-over, the unit install, the start -- and each must leave no identity
# file, no service and no registry row behind.
add_executable(test_node_cli tests/test_node_cli.cpp
               src/node/cli.cpp src/node/api_client.cpp src/node/json.cpp src/node/jobs.cpp
               src/node/identity.cpp src/node/gpu_probe.cpp src/node/sha256.cpp
               # for isServiceReachablePath: the rule that decides whether the unit can start at all
               src/node/service_systemd.cpp)
target_include_directories(test_node_cli PRIVATE src/node)
target_link_libraries(test_node_cli PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(test_node_cli PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_cli COMMAND test_node_cli)

# `brae run` against a stub control plane: argv, where the admin token comes from, what is actually sent, and
# what a failure looks like to someone at a terminal.
add_test(NAME brae_run
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/brae_run.sh $<TARGET_FILE:brae> python3)
set_tests_properties(brae_run PROPERTIES TIMEOUT 120)
set_tests_properties(node_cli PROPERTIES TIMEOUT 60)

# The real transport, against a deliberately awkward local server: a redirect that would leak the bearer token,
# a body that never ends, a server that stalls, a proxy's HTML error page. Guarded on libcurl so a machine
# without the headers still builds and tests everything else -- the agent's behaviour is covered above the
# transport (node_api), and only these properties need a real socket.
find_package(CURL QUIET)
if(CURL_FOUND)
    add_executable(test_node_http tests/test_node_http.cpp
                   src/node/http_curl.cpp src/node/api_client.cpp src/node/json.cpp src/node/jobs.cpp)
    target_include_directories(test_node_http PRIVATE src/node)
    target_link_libraries(test_node_http PRIVATE CURL::libcurl)
    set_target_properties(test_node_http PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME node_http
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/node_http_run.sh
                     $<TARGET_FILE:test_node_http>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/node_http_server.py)
    set_tests_properties(node_http PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 180)
else()
    message(STATUS "brae: libcurl not found -- the agent transport test is skipped "
                   "(install libcurl4-openssl-dev)")
endif()

# Solver selection itself: routes on controlDict `application`, refuses the solvers brae does not have, and
# catches an application/ddtSchemes contradiction. Dict-only fake cases, no mesh and no GPU.
add_test(NAME solver_dispatch
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/solver_dispatch.sh
                 $<TARGET_FILE:brae>
                 ${CMAKE_CURRENT_BINARY_DIR}/solver_dispatch_work)
set_tests_properties(solver_dispatch PROPERTIES TIMEOUT 120)

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
# demo/ is gitignored (local-only, exceeds GitHub's file-size limit), so this
# target is built ONLY when the source is present -- CI (no demo/) skips it.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo/amgpcg/trace_brae.cu")
    add_executable(trace_brae demo/amgpcg/trace_brae.cu)
    target_include_directories(trace_brae PRIVATE ${BRAE_HEADER_DIRS})
    target_link_libraries(trace_brae PRIVATE brae_core)
endif()

# ---------------------------------------------------------------------------
# ACPP backend routing: retrofit every .cu-source target (LANGUAGE -> CXX + AdaptiveCpp's compiler-launch rule).
# Test targets aren't ACPP-audited -- build a specific target (`--target brae`), not `all`.
# ---------------------------------------------------------------------------
if(BRAE_BACKEND STREQUAL "ACPP")
    function(brae_route_target_through_acpp _target)
        get_target_property(_srcs ${_target} SOURCES)
        if(NOT _srcs)
            return()
        endif()
        set(_has_cuda_src FALSE)
        foreach(_src ${_srcs})
            if(_src MATCHES "\\.cuh?$")
                set(_has_cuda_src TRUE)
                set_source_files_properties("${_src}" PROPERTIES LANGUAGE CXX)
            endif()
        endforeach()
        if(_has_cuda_src)
            set_target_properties(${_target} PROPERTIES
                RULE_LAUNCH_COMPILE "${ACPP_COMPILER_LAUNCH_RULE}"
                RULE_LAUNCH_LINK    "${ACPP_COMPILER_LAUNCH_RULE}")
            target_link_libraries(${_target} PRIVATE AdaptiveCpp::acpp-rt)
            target_compile_definitions(${_target} PRIVATE BRAE_ACPP)
            # Force-includes acpp_pcuda_compat.cuh (cuda_runtime.h + the nvcc-isms PCUDA doesn't predefine).
            # One -include, not two -- CMake's Makefile generator collapses a second one into the first.
            target_compile_options(${_target} PRIVATE
                -include "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/acpp_pcuda_compat.cuh")
        endif()
    endfunction()

    get_property(_brae_all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_t ${_brae_all_targets})
        get_target_property(_t_type ${_t} TYPE)
        if(_t_type STREQUAL "EXECUTABLE" OR _t_type STREQUAL "STATIC_LIBRARY")
            brae_route_target_through_acpp(${_t})
        endif()
    endforeach()
endif()

# ---- install ---------------------------------------------------------------------------------------------------
#
#   sudo cmake --install build
#
# All of them, together. Installing by hand cost three separate outages in one week: an agent without the
# telemetry it was supposed to send, a `brae` that could not find `brae_pimpleFoam` beside it, and a `brae` that
# had never heard of the `job` subcommand. Every one of them looked like a server bug from the outside, because
# the binary that ran was older than the code that was tested.
#
# They travel as a set on purpose: `brae` dispatches a case to `brae_<solver>` as its own sibling, and
# `brae-agent` execs `brae` the same way, so a mixed-version bin directory fails at the moment work arrives
# rather than at install time.
# Keep the library search path on the way out. CMake strips the build RPATH at install time and replaces it
# with INSTALL_RPATH, which is empty by default -- so the first `cmake --install` produced binaries that could
# not find libnvshmem_host.so.3 and failed for every user, not only under sudo. USE_LINK_PATH carries the
# directories of the libraries actually linked (NVSHMEM, OpenMPI) into the installed binary.
foreach(exe IN ITEMS brae brae-agent brae_pimpleFoam)
    if(TARGET ${exe})
        set_target_properties(${exe} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
        install(TARGETS ${exe} RUNTIME DESTINATION bin)
    endif()
endforeach()
