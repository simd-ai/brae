# Every external dependency, found once in the root scope so that the targets of src/ and tests/ see the same
# imported targets, variables and CMAKE_BUILD_RPATH. Included by the root CMakeLists.txt.

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

# CUDAToolkit: the NVRTC runtime compiler (CUDA::nvrtc) + the driver API (CUDA::cuda_driver) for the device-resident
# runtime-compiled boundary conditions (device_coded_bc.cu: nvrtc -> PTX -> cuModuleLoadData -> cuLaunchKernel).
find_package(CUDAToolkit REQUIRED)

# SCOTCH, cell-graph partitioner (the decomposePar-equivalent), matching OpenFOAM.
find_path(SCOTCH_INCLUDE_DIR NAMES scotch.h PATH_SUFFIXES scotch REQUIRED)
find_library(SCOTCH_LIBRARY NAMES scotch REQUIRED)
find_library(SCOTCH_ERR_LIBRARY NAMES scotcherr scotcherrexit REQUIRED)
message(STATUS "cf SCOTCH: ${SCOTCH_LIBRARY} (hdr ${SCOTCH_INCLUDE_DIR})")

find_package(Threads REQUIRED)          # brae-agent: the executor runs a job on a worker thread so heartbeats continue
find_package(CURL QUIET)

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
        # Find libnvshmem_host.so.3 and the dlopen'd MPI bootstrap plugin at runtime.
        list(APPEND CMAKE_BUILD_RPATH ${NVSHMEM_LIB_DIR})
    else()
        message(WARNING "cf NVSHMEM requested but not found; building the MPI backend only (single-GPU)")
        set(BRAE_WITH_NVSHMEM OFF)
    endif()
endif()
