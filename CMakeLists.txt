cmake_minimum_required(VERSION 3.24)

# Target NVIDIA GB10 / Grace Blackwell (sm_121). Override with
# -DCMAKE_CUDA_ARCHITECTURES=90 etc. on older toolkits. Phase 1 is host code in .cu
# files (compiled by nvcc); the arch matters once kernels land in the G-phases.
set(BRAE_DEFAULT_CUDA_ARCH 121)
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES ${BRAE_DEFAULT_CUDA_ARCH}
        CACHE STRING "CUDA architectures cf is built for" FORCE)
endif()

project(brae VERSION 0.3.1 LANGUAGES CXX CUDA)

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

# CUDAToolkit: the NVRTC runtime compiler (CUDA::nvrtc) + the driver API (CUDA::cuda_driver) for the device-resident
# runtime-compiled boundary conditions (device_coded_bc.cu: nvrtc -> PTX -> cuModuleLoadData -> cuLaunchKernel).
find_package(CUDAToolkit REQUIRED)

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
    src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nut_wall_function.cu
    src/finiteVolume/finiteVolume/fvc.cu
    src/finiteVolume/finiteVolume/fvm.cu
    src/OpenFOAM/matrices/ldu_spmv.cu
    src/OpenFOAM/matrices/pcg.cu
    src/OpenFOAM/matrices/gamg.cu
    src/OpenFOAM/matrices/pbicgstab.cu
    src/matrices/lduMatrix/lduMatrix/blas1.cu
    src/matrices/lduMatrix/lduMatrix/reductions.cu
    src/matrices/lduMatrix/lduMatrix/device_spmv.cu
    src/matrices/lduMatrix/preconditioners/DILUPreconditioner/device_dilu.cu
    src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/realizableKE/realizableKE_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu
    src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras_cpp.cu
    src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cu
    src/finiteVolume/cfdTools/general/fvOptions/rotorDiskSource_cpp.cu
    src/finiteVolume/cfdTools/general/fvOptions/actuationDiskSource_cpp.cu
    src/finiteVolume/cfdTools/general/bound/bound_cpp.cu
    src/finiteVolume/cfdTools/general/MRF/MRF_cpp.cu
    src/finiteVolume/cfdTools/general/MRF/device_MRF.cu
    src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cu
    src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellLimitedGrad_cpp.cu
    src/applications/solvers/simpleFoam/UEqn_cpp.cu
    src/applications/solvers/simpleFoam/UEqn.cu
    src/applications/solvers/simpleFoam/pEqn.cu
    src/applications/solvers/simpleFoam/simpleFoam.cu
    src/applications/solvers/simpleFoam/simpleFoamV2.cu
    src/applications/solvers/simpleFoam/pEqn_cpp.cu
    src/applications/solvers/simpleFoam/createFields_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu
    src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu
    src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu
    src/applications/solvers/rhoSimpleFoam/rhoPcEqn.cu
    src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam.cu
    src/applications/solvers/rhoSimpleFoam/rhoCreateFields.cu
    src/applications/solvers/rhoSimpleFoam/rhoThermoDevice.cu
    src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoPEqn_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoPcEqn_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoSimpleFoamDriver_cpp.cu
    src/applications/solvers/rhoSimpleFoam/rhoTurbulenceHook.cu
    src/applications/solvers/rhoSimpleFoam/rhoSimpleFoamDriver.cu
    src/applications/solvers/simpleFoam/simpleFoam_cpp.cu
    src/finiteVolume/cfdTools/general/solutionControl/simpleControl/simpleControl_cpp.cu
    src/matrices/lduMatrix/solvers/PCG/device_pcg.cu
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
    src/cuda/device_coded_bc.cu
    src/cuda/device_energy.cu
    src/thermophysicalModels/device_thermo.cu
    src/finiteVolume/cfdTools/MRF/device_mrf.cu
    src/finiteVolume/cfdTools/fvOptions/device_fvoptions.cu
    src/finiteVolume/cfdTools/fvOptions/rotor_disk.cu
    src/finiteVolume/cfdTools/fvOptions/actuation_disk.cu
    src/finiteVolume/fvMatrices/cyclicAMI/cyclicAMI_cpp.cu
    src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg.cu
    src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_cache.cu
    src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_coarse.cu
    src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_smoothers.cu
    src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_gauss_seidel.cu
    src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_vcycle.cu
    src/matrices/lduMatrix/solvers/PCG/device_amg_pcg.cu
    src/cuda/device_cluster.cu
    src/cuda/device_cyclic.cu
    src/cuda/device_ami.cu
    src/applications/solvers/simpleFoam/device_simple_foam.cu
)
target_include_directories(brae_core PUBLIC ${BRAE_HEADER_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${SCOTCH_INCLUDE_DIR})
# Link MPI libraries directly (not the imported target) so its CXX compile flags do not
# reach nvcc, which would reject them.
target_link_libraries(brae_core PUBLIC ${MPI_CXX_LIBRARIES} ${SCOTCH_LIBRARY} ${SCOTCH_ERR_LIBRARY} ZLIB::ZLIB CUDA::nvrtc CUDA::cuda_driver)
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

# The _cpp reference for UEqn.H, validated by decomposition against the same OpenFOAM momentum dump:
# core vs OpenFOAM, then the dev2 term's effect isolated, then relaxation, then the MRF/fvOptions refusals.
# The CUDA UEqn against the _cpp reference, field by field (decomposed: diag/upper/lower, the three
# sources, and every boundary coefficient on every patch). Laminar and turbulent, because with constant
# nuEff a per-face-diffusivity error cannot show up.
add_executable(test_ueqn_cuda tests/test_ueqn_cuda.cu)
target_include_directories(test_ueqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ueqn_cuda PRIVATE brae_core)
add_test(NAME ueqn_cuda
         COMMAND test_ueqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME ueqn_cuda_turbulent
         COMMAND test_ueqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

# One binary, one scheme each: CUDA against the _cpp reference at MATRIX granularity for every ported
# div(phi,U) scheme. The tolerances differ by scheme on purpose -- the r-ratio limiters divide by a face
# difference that approaches zero and land at 1e-12/1e-13, while linearUpwindV and LUST hit 1e-16.
foreach(_sch limitedLinear limitedLinearV LUST linearUpwindV)
    add_test(NAME ueqn_cuda_scheme_${_sch}
             COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_SCHEME=${_sch}
                     $<TARGET_FILE:test_ueqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
endforeach()

# CUDA rhoUEqn against the _cpp reference at MATRIX granularity -- diag, upper, lower, the three
# sources and every boundary coefficient, not a folded single number. The reference is itself gated
# against OpenFOAM's own dump by tests/rho_ueqn_vs_openfoam.sh, so this closes OpenFOAM -> _cpp -> CUDA.
#
# The fixtures are the incompressible ones because no committed case ships a rho field -- the compressible
# _cpp gates build their state by running real OpenFOAM. The gate SYNTHESIZES a non-uniform rho and
# asserts it varies, which is the point: with a uniform rho, mu_eff is a constant multiple of nu_eff and a
# kernel that dropped the rho weighting from one term would still agree on a rescaled matrix.
# CUDA rhoPEqn against the _cpp reference, stage by stage and BOTH BRANCHES. simple.transonic() selects
# two different equations -- the transonic one adds fvm::div(phid,p) and calls pEqn.relax(), the subsonic
# one runs adjustPhi and neither -- so both are registered. Testing one would leave the other asserted by
# nothing. rho and psi are synthesized non-uniform for the reason the gate's header gives.
# CUDA rhoEEqn against the _cpp reference, stage by stage and BOTH ENERGY VARIABLES. EEqn.H builds the
# kinetic-energy source as Ekp = 0.5|U|^2 + p/rho when he is `e` and as K = 0.5|U|^2 when he is `h` --
# different conservation laws, so both are registered. he, rho and alphaEff are synthesized non-uniform;
# alphaEff varies on the boundary INDEPENDENTLY of the cells so the patch-value rule is exercised.
# CUDA rhoPcEqn (SIMPLEC) against the _cpp reference, BOTH BRANCHES. It needs its own gate rather than
# riding on rho_peqn_cuda: rAtU = 1/(1/rAU - H1) is compared separately from rAU with a control asserting
# they differ, because if H1 came back zero every downstream number would still agree and the equation
# would be SIMPLE wearing SIMPLEC's name. The flux correction and the HbyA correction are asserted to move
# their fields independently -- they are one change, and keeping one without the other is worse than
# neither.
# The CUDA kEpsilon closure against the _cpp reference, stage by stage. The closure folds a production
# term, a wall treatment, two transport operators, a relaxation and two solves into k and epsilon, so one
# number over the solved fields cannot say which of them moved -- eps0/G0/isWallCell are compared on their
# own for exactly that reason. Three controls carry it: the MASS flux must not serve as divU (they differ
# only because rho varies here, which is why rho is synthesized varying rather than uniform), nut's
# boundary must be the PATCH value and not the owner cell's, and the production term must be load-bearing.
# A fourth runs correct() TWICE in one process: DeviceBuffer::resize takes from a pool that does not zero,
# and the first call is always right because the pool has nothing to hand back yet.
# The CUDA rhoSimpleFoam DRIVER against the _cpp driver, over MULTIPLE ITERATIONS -- which is the whole
# point. Every component it composes already has a per-stage gate at 1e-12 or better; what those cannot
# see is what the driver feeds them on the SECOND pass. The incompressible twin's AMG-cache defect was
# EXACT at iteration 1 and 1.3e-01 wrong at iteration 2 with every per-stage gate green, so a
# one-iteration driver test would have certified it. Laminar on purpose: the closure has its own device
# gate, and running it here would fold two more Krylov solves into a comparison about ORDER.
# The device projection of createFields against the host field set it projects. An upload is a copy, so
# the field checks assert EXACT equality -- a tolerance on a memcpy admits a defect rather than measuring
# one. What is actually at risk is the boundary-face ORDER, the padding of the faces DeviceMesh skips,
# which of the two masks answers which question, and whether the wall set is chosen by the BC or by the
# patch type; each has its own check, and the two a fixture can make vacuous say so instead of passing.
# STRUCTURAL measurements on OpenFOAM's angledDuctExplicitFixedCoeff tutorial -- the only case reachable
# from here that can measure two things this port had correct-by-construction and gated by nothing: the
# two boundary masks (it gives porosityWall `type slip` on U, and slip is the only patch class where
# `assignable` and `fixesValue && !inletOutlet` disagree), and the relax -> constrain -> setValues order
# (its fvOptions constrains k and epsilon on the porosity cellZone, whose own wall carries an
# epsilonWallFunction, so the two constraints overlap on every cell of that patch).
#
# NO CONVERGENCE BOUND. The end-to-end gate on this same tutorial is deliberately unregistered because a
# bound would have to accommodate an unexplained 9.43e-04; everything here is structural or a single
# assembly, so none of it inherits that question.
# EMPTY PATCHES MUST BE INERT. In OpenFOAM they cannot contribute to a surface sum because
# emptyFvPatchField is zero-sized; brae keeps the faces, so every summing operator must skip them
# explicitly. This asserts the PROPERTY -- poison an empty patch's values and require bit-identical
# output -- rather than a number, because a number would be satisfied by a mesh that happens to cancel.
# Measured before the fix: a cell's empty-face area vectors sum to 1.6e-16 of their magnitude on rhoBox
# and 2.4e-14 on pitzDaily, so the old behaviour was round-off that depended on the mesh being an exact
# extrusion. The control poisons a NON-empty patch and requires the answer to move.
add_executable(test_empty_patch_inert tests/test_empty_patch_inert.cu)
target_include_directories(test_empty_patch_inert PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_empty_patch_inert PRIVATE brae_core)
# DELIBERATELY NOT REGISTERED -- and the reason is measured, not assumed. Making empty patches inert is
# correct against OpenFOAM, and the operator-level change it causes is 7.886e-16 relative on pitzDailyFoam
# (the empty contribution to grad(p) is 4.562325e-12 against max|grad(p)| = 5.785680e+03). But a 1e-16
# perturbation is not nothing after 2000 SIMPLE iterations on a separating flow: applying it moved
# tutorial_backwardfacingstep2d's converged U from 3.482e-03 to 1.145e-02 against its 1e-02 bound --
# deterministic and reproducible to every printed digit, in a recirculation whose reattachment point is
# exactly the sensitive quantity. Registering this would therefore mean either shipping a red gate or
# loosening backwardFacingStep2D's bound to accommodate a change, which is the failure mode this port
# exists to avoid. The binary is built so the requirement stays executable and the measurement stays
# reproducible; enforcing it needs a convergence-robust comparison on that tutorial first.
#
# THAT COMPARISON NOW EXISTS (2026-09-03): backwardFacingStep2D is gated on a WINDOW MEAN, because the
# case limit-cycles and a single iterate was never a comparable quantity -- OpenFOAM moves U by 1.66e-02
# between two of its own iterates. A perturbation that shifts the phase moves the mean by ~5e-04, inside
# OpenFOAM's own noise floor. So this blocker is lifted in principle; registering the test still needs its
# own measurement through the windowed gate, which nothing here has done.

add_executable(test_rho_angledduct_structural tests/test_rho_angledduct_structural.cu)
target_include_directories(test_rho_angledduct_structural PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_angledduct_structural PRIVATE brae_core)
add_test(NAME rho_angledduct_structural
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_angledduct_structural.sh)
set_tests_properties(rho_angledduct_structural PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77)

add_executable(test_rho_createfields_cuda tests/test_rho_createfields_cuda.cu)
target_include_directories(test_rho_createfields_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_createfields_cuda PRIVATE brae_core)
# rhoBox is laminar and has neither inletOutlet nor a wall function; sbMatched has both, so between them
# every control below is exercised rather than reported vacuous.
add_test(NAME rho_createfields_cuda_laminar
         COMMAND test_rho_createfields_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/rhoBox 0.orig)
add_test(NAME rho_createfields_cuda_turbulent
         COMMAND test_rho_createfields_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/sbMatched 0.orig)

add_executable(test_rho_simple_step_cuda tests/test_rho_simple_step_cuda.cu)
target_include_directories(test_rho_simple_step_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_simple_step_cuda PRIVATE brae_core)
add_test(NAME rho_simple_step_cuda
         COMMAND test_rho_simple_step_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/rhoBox 0.orig 8)
add_test(NAME rho_step_cuda_refusals
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_step_cuda_refusals.sh)
set_tests_properties(rho_step_cuda_refusals PROPERTIES SKIP_RETURN_CODE 77)
# The BOUNDARY arm. rhoBox carries only fixedValue, zeroGradient, noSlip and empty, so the whole
# updateCoeffs() block at the top of the step is a no-op there and the arm above cannot see it at all --
# which is exactly why the device driver went without one. sbMatched is the only compressible fixture
# with four inletOutlet patches and a flowRateInletVelocity inlet. ONE iteration and no forced limiters;
# the reasons are in the test's own header. Measured: Ux 4.496e-12, and 1.276e-02 with the driver's
# updateBoundaryCoeffs call removed.
add_test(NAME rho_simple_step_cuda_boundary
         COMMAND test_rho_simple_step_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/sbMatched 0.orig --boundary)
# DEVICE-RESIDENT arms: the same driver with thermo.correct(), thermo.rho() and the effective transport
# run on the GPU instead of round-tripping the field set to the _cpp reference and back. Two fixtures,
# because between them they take BOTH branches of every switch in the device thermo:
#   rhoBox     hePsiThermo  + const      + sensibleEnthalpy       -> rho = p*psi, alpha = mu/Pr, CpByCpv 1
#   sbMatched  heRhoThermo  + sutherland + sensibleInternalEnergy -> rho = stored, Eucken kappa/Cp, gamma
# Each carries an oracle control that recomputes the thermo relations on the host from the device's own
# he and p, so the arm cannot pass by two implementations agreeing on nothing.
add_test(NAME rho_simple_step_cuda_device_thermo
         COMMAND test_rho_simple_step_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/rhoBox 0.orig 8 --device-thermo)
add_test(NAME rho_simple_step_cuda_device_thermo_rho
         COMMAND test_rho_simple_step_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/sbMatched 0.orig --boundary --device-thermo)

# THE MANIFEST MUST MATCH ITS GENERATOR. tools/of_manifest.py already had a --check mode that reports
# drift, and nothing ran it -- so manifest/rhoSimpleFoam.yaml sat stale against the generator while
# still being the file PORT.md calls authoritative. It was carrying `U 1.08e-04 (gated at 2e-3)` for a
# gate whose bound is 5e-4 and which now holds eight fields. A check that exists but is registered
# nowhere is not a check; this is the registration.
# deviceAdjustPhi against OpenFOAM's own two guards. No fixture: the guards are arithmetic on three flux
# sums, so the inputs are constructed exactly -- including the cases a real mesh reaches only by accident.
add_executable(test_adjust_phi_guards tests/test_adjust_phi_guards.cu)
target_include_directories(test_adjust_phi_guards PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_adjust_phi_guards PRIVATE brae_core)
add_test(NAME adjust_phi_guards COMMAND test_adjust_phi_guards)

add_executable(test_powerlaw_dict_guards tests/test_powerlaw_dict_guards.cu)
target_include_directories(test_powerlaw_dict_guards PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_powerlaw_dict_guards PRIVATE brae_core)
add_test(NAME powerlaw_dict_guards COMMAND test_powerlaw_dict_guards)

add_executable(test_frozen_bc_guard tests/test_frozen_bc_guard.cu)
target_include_directories(test_frozen_bc_guard PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_frozen_bc_guard PRIVATE brae_core)
add_test(NAME frozen_bc_guard COMMAND test_frozen_bc_guard)
add_test(NAME frozen_bc_refusals COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/frozen_bc_refusals.sh)
set_tests_properties(frozen_bc_refusals PROPERTIES SKIP_RETURN_CODE 77)

# Audit finding #14: the algebraic-LES path honoured only nutUSpalding while printing any other nut
# wall function as honoured, labelled "kEpsilon" whatever the model. Non-Spalding wall functions on
# LES now refuse by name, and every wall-fn print carries the model word the case names.
add_test(NAME les_wallfunction_refusals COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/les_wallfunction_refusals.sh)
set_tests_properties(les_wallfunction_refusals PROPERTIES SKIP_RETURN_CODE 77)

# Audit finding #15: the SA device path hard-forced Spalding whatever 0/nut asked for --
# bump2D:SpalartAllmaras ships nutLowReWallFunction (calcNut() = Zero on every model) and got a Newton
# uTau inside a green gate. The BC selects now: Spalding/LowRe honoured, k-based family and plain wall
# BCs refused by name, two different wall functions refused.
add_test(NAME sa_wallfunction_refusals COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/sa_wallfunction_refusals.sh)
set_tests_properties(sa_wallfunction_refusals PROPERTIES SKIP_RETURN_CODE 77)

# Audit finding #16: the turbulence-setup guards compared boundaryField keys to patch names exactly, so
# regex/group keys (bfs2D writes every wall BC as "(upperWall|lowerWall)") disarmed them silently while
# buildField resolved the same keys correctly. The guards resolve through patchesResolvingTo now; two
# arms are discriminating (they ran silently before the fix).
add_test(NAME regex_patchkey_guards COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/regex_patchkey_guards.sh)
set_tests_properties(regex_patchkey_guards PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_v2_laplacian_parse tests/test_v2_laplacian_parse.cu)
target_include_directories(test_v2_laplacian_parse PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_v2_laplacian_parse PRIVATE brae_core)
add_test(NAME v2_laplacian_parse COMMAND test_v2_laplacian_parse)

add_executable(test_tvm_mapping tests/test_tvm_mapping.cu)
target_include_directories(test_tvm_mapping PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_tvm_mapping PRIVATE brae_core)
add_test(NAME tvm_mapping COMMAND test_tvm_mapping)

add_test(NAME manifest_rhosimplefoam_not_drifted
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/of_manifest.py rhoSimpleFoam
                 --check ${CMAKE_CURRENT_SOURCE_DIR}/manifest/rhoSimpleFoam.yaml)

# The simpleFoam twin existed only as a tool flag, and its manifest drifted for weeks: sixteen whole
# entries and four long validation/note blocks were hand-added to the YAML while the tool's curated
# table (where they are supposed to live) fell behind. Reconciled 2026-08-31; this test is what keeps
# the two from diverging again.
add_test(NAME manifest_simplefoam_not_drifted
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/of_manifest.py simpleFoam
                 --check ${CMAKE_CURRENT_SOURCE_DIR}/manifest/simpleFoam.yaml)

add_executable(test_rho_kepsilon_cuda tests/test_rho_kepsilon_cuda.cu)
target_include_directories(test_rho_kepsilon_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_kepsilon_cuda PRIVATE brae_core)
add_test(NAME rho_kepsilon_cuda
         COMMAND test_rho_kepsilon_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_pceqn_cuda tests/test_rho_pceqn_cuda.cu)
target_include_directories(test_rho_pceqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_pceqn_cuda PRIVATE brae_core)
add_test(NAME rho_pceqn_cuda_subsonic
         COMMAND test_rho_pceqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_pceqn_cuda_subsonic_turbulent
         COMMAND test_rho_pceqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
add_test(NAME rho_pceqn_cuda_transonic
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_pceqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_pceqn_cuda_transonic_turbulent
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_pceqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_eeqn_cuda tests/test_rho_eeqn_cuda.cu)
target_include_directories(test_rho_eeqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_eeqn_cuda PRIVATE brae_core)
add_test(NAME rho_eeqn_cuda_e
         COMMAND test_rho_eeqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282)
add_test(NAME rho_eeqn_cuda_e_turbulent
         COMMAND test_rho_eeqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
add_test(NAME rho_eeqn_cuda_h
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_ENERGY=h
                 $<TARGET_FILE:test_rho_eeqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_peqn_cuda tests/test_rho_peqn_cuda.cu)
target_include_directories(test_rho_peqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_peqn_cuda PRIVATE brae_core)
add_test(NAME rho_peqn_cuda_subsonic
         COMMAND test_rho_peqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_peqn_cuda_subsonic_turbulent
         COMMAND test_rho_peqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
add_test(NAME rho_peqn_cuda_transonic
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_peqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_peqn_cuda_transonic_turbulent
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_TRANSONIC=1
                 $<TARGET_FILE:test_rho_peqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_rho_ueqn_cuda tests/test_rho_ueqn_cuda.cu)
target_include_directories(test_rho_ueqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_ueqn_cuda PRIVATE brae_core)
add_test(NAME rho_ueqn_cuda
         COMMAND test_rho_ueqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME rho_ueqn_cuda_turbulent
         COMMAND test_rho_ueqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

# One binary, one scheme each. The tolerances differ by scheme on purpose -- the r-ratio limiters divide
# by a face difference that approaches zero, while linearUpwindV and LUST reach 1e-16.
# `grad(U) cellLimited Gauss linear 1` on the 2D pitzDailyTurb, which is the ONLY registered arm that
# makes the EMPTY-patch question visible. emptyFvPatchField is zero-sized in OpenFOAM so those faces do
# not exist there; brae keeps them and must skip them in both the limiter's range and its face loop, and
# in the face loop it is not harmless -- Cf - C points out of the 2D plane, so the extrapolate is
# round-off and the ratio against maxDelta can clamp the limiter below what any real face asks for.
add_test(NAME rho_ueqn_cuda_cell_limited
         COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_GRADLIMIT=1.0
                 $<TARGET_FILE:test_rho_ueqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

foreach(_sch upwind limitedLinear limitedLinearV LUST linearUpwindV)
    add_test(NAME rho_ueqn_cuda_scheme_${_sch}
             COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_SCHEME=${_sch}
                     $<TARGET_FILE:test_rho_ueqn_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
endforeach()

add_executable(test_ueqn_cpp tests/test_ueqn_cpp.cu)
target_include_directories(test_ueqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ueqn_cpp PRIVATE brae_core)

add_executable(test_komegasst_cpp tests/test_komegasst_cpp.cu)
target_include_directories(test_komegasst_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_komegasst_cpp PRIVATE brae_core)

add_executable(test_realizablke_cpp tests/test_realizablke_cpp.cu)
target_include_directories(test_realizablke_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_realizablke_cpp PRIVATE brae_core)

add_executable(kepsilon_probe tests/kepsilon_probe.cu)
target_include_directories(kepsilon_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(kepsilon_probe PRIVATE brae_core)

add_executable(resid_probe tests/resid_probe.cu)
target_include_directories(resid_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(resid_probe PRIVATE brae_core)

add_executable(test_fvoptions_cpp tests/test_fvoptions_cpp.cu)
target_include_directories(test_fvoptions_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvoptions_cpp PRIVATE brae_core)

add_executable(test_limitedschemes_cpp tests/test_limitedschemes_cpp.cu)
target_include_directories(test_limitedschemes_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_limitedschemes_cpp PRIVATE brae_core)
add_test(NAME ueqn_cpp
         COMMAND test_ueqn_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

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

# The _cpp reference for pEqn.H, stage by stage: rAU/HbyA vs OpenFOAM's A()/H() (ops.dat), the pressure
# Laplacian vs peqn.dat, then setReference / correctFlux / relax asserted analytically, plus the refusals.
# The CUDA pEqn against the _cpp reference, stage by stage. Laminar and turbulent.
add_executable(test_peqn_cuda tests/test_peqn_cuda.cu)
target_include_directories(test_peqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_peqn_cuda PRIVATE brae_core)
add_test(NAME peqn_cuda
         COMMAND test_peqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME peqn_cuda_turbulent
         COMMAND test_peqn_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_peqn_cpp tests/test_peqn_cpp.cu)
target_include_directories(test_peqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_peqn_cpp PRIVATE brae_core)
add_test(NAME peqn_cpp
         COMMAND test_peqn_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_peqn_matrix tests/test_peqn_matrix.cu)
target_include_directories(test_peqn_matrix PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_peqn_matrix PRIVATE brae_core)
add_test(NAME peqn_matrix
         COMMAND test_peqn_matrix ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym)

# END-TO-END gate for the rebuilt simpleFoam: one SIMPLE iteration composed of the _cpp components,
# against OpenFOAM's own dumpSimpleStep output. Proves the composition, not just the parts.
# END-TO-END: the CUDA driver against the _cpp driver, one iteration.
# Diagnostic (not a ctest): both drivers, N iterations, side-by-side residuals + divergence trace.
add_executable(diag_simple_loop tests/diag_simple_loop.cu)
target_include_directories(diag_simple_loop PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(diag_simple_loop PRIVATE brae_core)

add_executable(test_simple_step_cuda tests/test_simple_step_cuda.cu)
target_include_directories(test_simple_step_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_step_cuda PRIVATE brae_core)
add_test(NAME simple_step_cuda
         COMMAND test_simple_step_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpSimple 282 laminar)
# 10 iterations: the CUDA driver must still track the _cpp driver. Pins the AMG-hierarchy-vs-transient-
# pressure-buffer bug, which was exact at iteration 1 and wrong by 1.3e-01 at iteration 2.
add_test(NAME simple_step_cuda_loop
         COMMAND test_simple_step_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpSimple 282 laminar 10)

add_test(NAME simple_step_cuda_turbulent
         COMMAND test_simple_step_cuda ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_simple_step_cpp tests/test_simple_step_cpp.cu)
target_include_directories(test_simple_step_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_step_cpp PRIVATE brae_core)
add_test(NAME simple_step_cpp
         COMMAND test_simple_step_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpSimple)

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

# DETERMINISM GATE. Two runs of the same binary on the same case must agree bit for bit. This is a
# prerequisite for the whole _cpp-vs-CUDA validation plan: a defect smaller than the run-to-run noise floor
# cannot be detected, and the floor here used to be 3.6e-02 after 20 iterations. Two cases, because the
# turbulence wall functions were one of the three scatter sites and kEpsilon/kOmegaSST take different ones.
# SKIP_RETURN_CODE 77 lets it skip cleanly when no GPU binary is present.
# DISPATCH GATE for the rebuilt simpleFoam: off changes nothing, on+supported runs, on+unsupported
# REFUSES with the reason. A path that quietly degrades is indistinguishable from a correct one.
# A regex-keyed turbulence residualControl must actually bind -- see the script for why this exists and
# for what it is not (the defect it was written to chase turned out to be a symptom of T3A diverging).
add_test(NAME simple_residualcontrol_regex
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/simple_residualcontrol_regex.sh)
set_tests_properties(simple_residualcontrol_regex PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1800)

add_test(NAME simplefoam_v2_dispatch
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/simplefoam_v2_dispatch.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(simplefoam_v2_dispatch PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# CONVERGENCE GATE: run the rebuilt path to convergence and compare with OpenFOAM's own converged fields
# (validation/pitzDailyTurb/1576, genuine simpleFoam v2412 output). The bounds are what brae's established
# solver achieves against the same reference in simple_turbulent_full. ~90 s.
add_test(NAME simplefoam_v2_convergence
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/simplefoam_v2_convergence.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
set_tests_properties(simplefoam_v2_convergence PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 600)

# NON-ORTHOGONAL CORRECTION vs real OpenFOAM on a genuinely non-orthogonal mesh, with a control requiring
# that omitting the term is clearly worse -- on a near-orthogonal mesh this gate would pass with the term
# deleted. Skips (77) where OpenFOAM is not installed.
# STOCK pitzDaily end to end: SIMPLEC + linearUpwind + bounded + corrected + kEpsilon, unmodified, vs
# real OpenFOAM. Also asserts SIMPLEC and plain SIMPLE reach the SAME fixed point -- the control that
# exposed the missing fvMatrix::faceFluxCorrection.
add_test(NAME stock_pitzdaily_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/stock_pitzdaily_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(stock_pitzdaily_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# The k/epsilon PATCH diffusivity: OF's `calculated` nut patches carry Cmu*k_b^2/eps_b, not the cell value,
# and DkEff(patchi)/DepsilonEff(patchi) are built from it. Started from OpenFOAM's own converged fields so
# the initial residual measures the discretisation directly; needs no OpenFOAM install, only its log.
add_test(NAME boundary_nut_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/boundary_nut_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb)
set_tests_properties(boundary_nut_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# kOmegaSST on the _cpp path vs real OpenFOAM. The fast entry probes one iteration from OpenFOAM's own
# converged state and carries its own upwind negative control; the slow one runs the case end to end from
# 0/, which is what exposed the omega blow-up that the probe could not see.
add_test(NAME sst_cpp_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/sst_cpp_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailySST)
set_tests_properties(sst_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME sst_cpp_full
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/sst_cpp_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailySST full)
set_tests_properties(sst_cpp_full PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1200)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one module at a time. The fast entry
# also asserts brae RESOLVES div(phi,k) to `bounded limitedLinear` rather than silently running upwind, and
# is what catches the wall mask: keying it on epsilonWallFunction left every kOmegaSST case with no wall
# faces at all, measuring U 48x and omega 6.8x off OpenFOAM.
add_test(NAME sst_cuda_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/sst_cuda_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailySST)
set_tests_properties(sst_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME sst_cuda_full
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/sst_cuda_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailySST full)
set_tests_properties(sst_cuda_full PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# MRF end to end on the _cpp path vs real OpenFOAM, with the control that makes it mean something: with
# the zones dropped this case has no driving force at all and every field reads 1.000 against OpenFOAM.
add_test(NAME mrf_cpp_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/mrf_cpp_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/mixerVessel2D)
set_tests_properties(mrf_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one hook at a time. Its control
# removes constant/MRFProperties, which exercises the real dictionary path rather than a debug switch.
add_test(NAME mrf_cuda_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/mrf_cuda_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/mixerVessel2D)
set_tests_properties(mrf_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

# SpalartAllmaras end to end on the _cpp path vs real OpenFOAM, with a laminar control. This is the case
# that exposed nut's wall value being taken from the nut_ = nuTilda*fv1 assignment instead of
# nutUSpaldingWallFunction -- see the script header for the three defects it cost.
add_test(NAME sa_cpp_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/sa_cpp_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(sa_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one module at a time. Its cold start
# is what the fixed-point comparison could not see: the two momentum assemblies are bit-identical, and the
# whole gap was the Spalding wall Newton being cold-started instead of seeded from nut_ -- see the header.
add_test(NAME sa_cuda_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/sa_cuda_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(sa_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# cellLimited Gauss linear <k> vs real OpenFOAM, on BOTH paths, with an unlimited control that must be
# markedly worse. This is what windAroundBuildings was refused for: linearUpwind's correction is built
# from the gradient the scheme NAMES, and it does not vanish at convergence.
add_test(NAME celllimited_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/celllimited_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/windAroundBuildingsBox)
set_tests_properties(celllimited_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# rotorDiskSource vs OpenFOAM's OWN reported drag/lift/AOA on the real 71734-cell tutorial mesh, plus a
# device-vs-_cpp check. The device implementation predated any test; this is the first thing to check it.
add_test(NAME rotordisk_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rotordisk_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/rotorDisk)
set_tests_properties(rotordisk_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# fv::actuationDiskSource (Froude) against the Uref/a/thrust OpenFOAM writes for itself, evaluated at the
# field OpenFOAM computed that row FROM -- which makes it exact to 13 digits rather than approximate.
add_test(NAME actuationdisk_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/actuationdisk_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/turbineSiting 600)
set_tests_properties(actuationdisk_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The SAME case through the CUDA V2 driver, end to end against real OpenFOAM at DEEP convergence. This is
# what pins the SIGN the thrust is applied with, which the .dat oracle above cannot see: it compares
# numbers computed from a given velocity, and those are identical whichever way the disk pushes. Its
# control is the same run with the turbines removed.
add_test(NAME turbinesiting_cuda_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/turbinesiting_cuda_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/turbineSiting)
set_tests_properties(turbinesiting_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

# OpenFOAM's fusedGauss scheme family (pitzDaily_fused). It is Gauss written without field temporaries,
# so brae reads it AS Gauss and announces that; this gate measures the equivalence from OpenFOAM ITSELF
# (its own two spellings of the same case), asserts brae's rewrite is bit-for-bit a rename, and runs the
# case end to end. Its control forces div(phi,U) to upwind.
add_test(NAME fusedgauss_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/fusedgauss_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(fusedgauss_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# kOmegaSSTLM (Langtry-Menter transition) on the _cpp REFERENCE, run END TO END on the real T3A tutorial
# against real OpenFOAM. Its control is the same case with the transition model removed -- 167x on U and
# 437x on k -- which is what makes the bounds above mean something.
add_test(NAME lm_cpp_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/lm_cpp_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/T3A 269 500)
set_tests_properties(lm_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one module at a time. The module
# comparison (lm_cuda_probe: Fthetat, both reaction preps and gammaIntEff to 1e-12) runs FIRST inside the
# gate, so an end-to-end agreement that came from cancellation cannot pass it. Control: plain kOmegaSST.
add_test(NAME lm_cuda_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/lm_cuda_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/T3A 269)
set_tests_properties(lm_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# turbulence->validate() (simpleFoam.C:92) is correctNut(), so OpenFOAM's FIRST momentum assembly runs on
# nut rebuilt from the initial k and omega and not on the case's `nut uniform 0`. Oracle: dumpSimpleFoam's
# stage_nuEff/stage_UDiag0/stage_rAU at iteration 1. Control: the file's own nut, which must disagree --
# it is the laminar first iteration this used to run.
# fvMatrix<vector>::solveSegregated skips the components polyMesh::solutionD() knocks out
# (fvMatrixSolve.C:164) and residualControl compares cmptMax over what it stored (solutionControl.C:232).
# The V2 step solved all three and reported component 0. Oracle: OpenFOAM's own log, one iteration from
# its own field; control: that fixture's Uy/Ux = 1.910, so component 0 reads 47.7% low; 3D arm proves the
# mask is derived and not hardcoded off.
# brae routes `smoothSolver` + `symGaussSeidel` to a MULTICOLOUR Gauss-Seidel sweep where
# symGaussSeidelSmoother.C sweeps in index order -- same stopping rule, less smoothing per sweep. It was
# announced nowhere and the run log said "as the case asks". Positive + two controls, no GPU and no OF.
# lduMatrix::solver::readControls reads minIter and maxIter from EACH FIELD'S own entry
# (lduMatrixSolver.C:190-208). The V2 driver read one maxIter from `p` and gave it to U as well, and
# minIter reached no solver at all. Oracle: OpenFOAM's own iteration counts under the same cap and floor.
# fvc::grad(U) resolves the gradSchemes `grad(U)` entry, and divDevReff (linearViscousStress.C:114) and
# every closure's strain and production are built from it. The V2 driver passed a literal 0.0 to the SST,
# let kEpsilon's and divDevReff's arguments default, and SA had no such parameter at all. Proximity gate:
# brae must land nearer OpenFOAM's LIMITED answer than its unlimited one, on the fields where brae is
# inside the two oracles' separation.
# endTime is an ABSOLUTE time (Time.C:785), deltaT is the step (Time.C:1067) and a time directory is
# named by the VALUE (Time.C:721-728). The V2 driver read endTime as an iteration count and never read
# deltaT. Oracle: OpenFOAM's own ordered list of time names on four shapes, one of them ratio 2.5 where a
# rounded quotient gives the wrong count.
# nSweeps (smoothSolver.C:78) is the number of smoothing sweeps between residual EVALUATIONS, and
# OpenFOAM advances its iteration count by nSweeps (:205), so every count it reports is a multiple.
# brae read the entry nowhere -- airFoil2D ships nSweeps 2 and brae reported odd counts. The negative
# fixed-sweep mode (:96-119) is refused: it deletes the convergence test and reports a residual of zero.
# linearUpwind, linearUpwindV and LUST all read the gradient name off div(phi,U)'''s own stream
# (linearUpwind.H:105, linearUpwindV.H:117-135, LUST.H). brae'''s regex demanded whitespace after
# `linearUpwind`, so V and LUST silently fell back to gradSchemes default, and it searched the whole
# divSchemes block rather than div(phi,U)'''s entry.
add_test(NAME linearupwindv_grad_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/linearupwindv_grad_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/windAroundBuildingsBox)
set_tests_properties(linearupwindv_grad_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1200)

add_test(NAME solver_nsweeps_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/solver_nsweeps_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/simpleBoxIO)
set_tests_properties(solver_nsweeps_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME timeloop_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/timeloop_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/simpleBoxIO)
set_tests_properties(timeloop_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME closure_gradlimit_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/closure_gradlimit_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/windAroundBuildingsBox)
set_tests_properties(closure_gradlimit_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1200)

add_test(NAME solver_iterbounds_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/solver_iterbounds_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/simpleBoxIO)
set_tests_properties(solver_iterbounds_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME gs_smoother_notice
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/gs_smoother_notice.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/T3A)
set_tests_properties(gs_smoother_notice PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

add_test(NAME uresidual_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/uresidual_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/simpleBoxIO
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/cav3d_of)
set_tests_properties(uresidual_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME validate_nut_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_nut_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/T3A)
set_tests_properties(validate_nut_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

# The cyclicAMI CUDA path against the _cpp REFERENCE, stage by stage: interpolation (scalar and rotated
# vector), face value, both matrix assemblies with their diagonals, the matrix action, the interface
# flux, UEqn.H()'s interface term and the interface's share of div(phi). Twelve checks at 1e-12; they
# pass bit-identically. It caught the reference's own assembleLaplacian sign on its first run.
add_test(NAME ami_cpp_vs_device
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/ami_cpp_vs_device.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pipeCyclicAMI 100
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pipeCyclicAMIConformal)
set_tests_properties(ami_cpp_vs_device PROPERTIES SKIP_RETURN_CODE 77)

# The DIV SCHEME's face weight at a coupled interface, on the real pipeCyclic tutorial against real
# OpenFOAM. brae assembled every cyclic and cyclicAMI face UPWIND whatever the case named -- a defect
# invisible to ami_cpp_vs_device above, because brae's device and its reference implemented the same wrong
# scheme and agreed to 1e-16 on it. Four runs, not one: brae and OpenFOAM under BOTH the limited and the
# upwind scheme, so matching OpenFOAM is checked against a case proven to discriminate the two. Plus a
# symmetry check that needs no reference at all -- the sector's transverse pressure force must vanish, and
# the upwind interface put it at 2.2e-02 of the axial force against 2.1e-06 now.
add_test(NAME coupledinterfacescheme_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/coupledinterfacescheme_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pipeCyclic)
set_tests_properties(coupledinterfacescheme_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The SAME gate on the other blend, `bounded Gauss linear`, which at a coupled face is the patch's own
# interpolation weight -- w ~ 0.5, the furthest a weight can get from upwind's 0 or 1, so this is the widest
# of the three gaps and the one a wrong interface assembly distorts most. OpenFOAM's implicitAMI is the only
# tutorial anywhere that pairs a coupled patch with `Gauss linear` on div(phi,U); it is laminar, so its
# bounds name only U and p and the gate checks the fields the case actually wrote.
add_test(NAME linearinterfacescheme_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/coupledinterfacescheme_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/implicitAMI U=4e-02,p=8e-02)
set_tests_properties(linearinterfacescheme_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# SIMPLEC (`consistent true`) across a coupled interface, on OpenFOAM's basic/simpleFoam/implicitAMI.
# fvMatrix::H1() sums boundaryCoeffs over every coupled patch and the SIMPLEC flux term is evaluated on
# coupled boundary faces like any other; brae did neither, so on a mesh coupled only by a cyclicAMI the
# whole SIMPLEC correction stopped at the interface. Nothing in validation/ combined SIMPLEC with a
# coupled interface before this case, which is why the path was unreachable. The sharp check seeds brae
# with OpenFOAM's own converged U, p and phi: at that state the true divergence is zero, so brae's
# continuity error measures its own inconsistency directly -- 1.3e-01 before, 1.7e-03 after, with the
# momentum residual at 2.8e-07 throughout to show the UEqn was never the term at fault.
add_test(NAME simplec_interface_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/simplec_interface_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/implicitAMI)
set_tests_properties(simplec_interface_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# Two stock simpleFoam tutorials that had no gate at all: they were run once, agreed, and nothing kept
# them agreeing. Both are read straight from the OpenFOAM installation, so neither needs a committed case.
# backwardFacingStep2D is the ONLY simpleFoam tutorial that names LUST, so it is the one thing standing
# behind brae's internal LUST weights. Each asserts that both codes reached a steady state before
# comparing -- squareBend reads U 2.759e-02 at its tutorial endTime and 1.773e-03 once both codes are on
# the same residual plateau, a factor of 16 from nothing but where the comparison was taken -- and each
# carries a control requiring the INITIAL field to fail the same bounds.
add_test(NAME tutorial_rotatingcylinders
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 rotatingCylinders U=2e-03,p=3e-03)
set_tests_properties(tutorial_rotatingcylinders PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# COMPARED AS A WINDOW MEAN, and the bounds below are ~2x what the mean actually achieves. This case has
# no steady state: OpenFOAM's own p residual sits at ~2.1e-02 for 20000 iterations and oscillates rather
# than falling, and against ITSELF 250 iterations apart it moves U by 1.66e-02 -- more than the 1e-02
# bound brae used to be held to at a single iterate. Averaging the last 21 writes (4000..6000) cuts
# OpenFOAM's own window-to-window variation to 6.05e-04 on U, and that mean is stationary (the window at
# 4000..6000 matches the one at 20000..29000 to 5.25e-04), which is why the window can be taken this
# early. Measured brae vs OpenFOAM: U 2.650e-03, p 3.363e-03, k 8.316e-03, omega 6.537e-04, nut 6.550e-03
# -- every one of them tighter than the single-iterate bound it replaces. See WINDOW in the script.
add_test(NAME tutorial_backwardfacingstep2d
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 backwardFacingStep2D U=6e-03,p=8e-03,k=2e-02,omega=2e-03,nut=1.5e-02
                 1e-04 6000/100 EMPTY 21)
set_tests_properties(tutorial_backwardfacingstep2d PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The MULTI-SETUP tutorials. bump2D and turbulentFlatPlate are not single cases: each ships a
# `setups.orig/` tree of one common skeleton plus a per-turbulence-model overlay, and turbulentFlatPlate
# additionally builds its mesh from a blockMeshDict.template expanded with a per-y+ grading -- so it is a
# matrix of (model x near-wall grading), not a list of cases. tests/tutorial_vs_openfoam.sh composes them
# exactly as their own Allrun does and names one as tutorial:setup[:grading].
#
# turbulentFlatPlate at y+ 1 is the tightest agreement of any tutorial gated here (U 1.9e-05, nut 3.8e-03),
# which is worth having precisely because bump2D at its own residualControl of 1e-05 is the loosest.
add_test(NAME tutorial_turbulentflatplate_komegasst
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 turbulentFlatPlate:kOmegaSST:2200 U=2e-04,p=3e-02,k=1e-03,omega=5e-04,nut=2e-02)
set_tests_properties(tutorial_turbulentflatplate_komegasst PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

add_test(NAME tutorial_bump2d_spalartallmaras
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 bump2D:SpalartAllmaras U=5e-03,p=5e-03,nut=1e-01,nuTilda=1e-01)
set_tests_properties(tutorial_bump2d_spalartallmaras PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

# ...and the setup brae must REFUSE. bump2D's third model is kEpsilonPhitF, which brae does not implement;
# running it as the kEpsilon it does have would be a wrong answer wearing the right label. The gate asserts
# brae exits nonzero AND names the model, so a refusal that says nothing useful fails too.
add_test(NAME tutorial_bump2d_kepsilonphitf_refused
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 bump2D:kEpsilonPhitF REFUSE:kEpsilonPhitF)
set_tests_properties(tutorial_bump2d_kepsilonphitf_refused PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# turbulentFlatPlate:kEpsilon at y+ ~ 1 -- the case that DIVERGED until the turbulence solve got the DILU
# preconditioner the case asks for. It is gated tightly on purpose: the failure it guards was not a wrong
# term but a preconditioner substitution, and the symptom of that returning is the run not finishing at
# all, so bounds near the measured 1.8e-05 on U cost nothing and say exactly what is expected.
add_test(NAME tutorial_turbulentflatplate_kepsilon
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 turbulentFlatPlate:kEpsilon:2200 U=2e-04,p=2e-03,k=3e-03,epsilon=2e-02,nut=2e-03)
set_tests_properties(tutorial_turbulentflatplate_kepsilon PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 3600)

# The two tutorials that had never been compared to OpenFOAM at all.
#
# motorBike is the big one -- 353700 cells from snappyHexMesh, kOmegaSST -- and the only tutorial here
# whose OpenFOAM side runs IN PARALLEL (decomposePar, snappyHexMesh and simpleFoam on 6 processors, then
# reconstructPar). Its ~5% on U is the LOOSEST agreement of any gated tutorial, and it is not a
# convergence artefact: run both codes 5x longer, to 2500, and the comparison is the same (U 5.58e-02
# against 5.00e-02, nut 9.56e-03 against 1.06e-02) while OpenFOAM's residual falls to 6.0e-05 and brae's
# to 1.4e-04. That is why the steady-state threshold is passed as 3e-04 here -- the extra iterations move
# the residual and not the answer -- and why the case is gated AS SHIPPED at its own endTime of 500.
# OpenFOAM's run is also potentialFoam-initialised and brae's is not, which is in the bounds too.
add_test(NAME tutorial_motorbike
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 motorBike U=8e-02,p=9e-02,k=1.7e-01,omega=1.0e-01,nut=2e-02 3e-04)
set_tests_properties(tutorial_motorbike PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 5400)

# pitzDailyExptInlet ships no Allrun at all (blockMesh + simpleFoam) and its residualControl is very loose
# -- p 1e-2, U 1e-3. That looseness is most of the gap: at the case's own settings brae reads U 2.866e-02
# and p 1.630e-01 against OpenFOAM, and with both codes driven to ~1e-9 the same comparison reads
# U 1.155e-02 and p 1.540e-02. Gated as shipped, with the bounds that implies.
add_test(NAME tutorial_pitzdailyexptinlet
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 pitzDailyExptInlet U=4e-02,p=2.5e-01,k=1.2e-01,epsilon=1.0e-01,nut=1.8e-01)
set_tests_properties(tutorial_pitzdailyexptinlet PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

# squareBend, gated at 8000 rather than its shipped endTime of 500. That is not padding: at 500 the
# comparison reads U 2.759e-02 and at 8000 -- where both codes sit on the same 2.5e-05 residual plateau --
# it reads 1.773e-03. Gating the shipped number would record where the run stopped as if it were a
# property of the solver. 112000 cells, kEpsilon, SIMPLEC, `Gauss limitedLinearV 1`.
add_test(NAME tutorial_squarebend
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 squareBend U=5e-03,p=5e-03,k=1.2e-02,epsilon=3e-02,nut=5e-03 1e-04 8000)
set_tests_properties(tutorial_squarebend PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 3600)

# bump2D:kOmegaSST. U, p, k and omega agree to 5.3e-04, 5.2e-04, 9.4e-03 and 1.1e-05; nut alone sits at
# 1.1e-01, and that is the case's own residualControl of 1e-05 rather than the closure -- drive both codes
# to 1e-08 and the same comparison reads U 1.495e-05 and nut 9.905e-03, a factor of 11 on nut from nothing
# but the stopping point. Gated as shipped, since 40000 iterations is not a routine gate, with a nut bound
# that says so.
add_test(NAME tutorial_bump2d_komegasst
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 bump2D:kOmegaSST U=1.5e-03,p=1.5e-03,k=2.5e-02,omega=4e-05,nut=2.2e-01)
set_tests_properties(tutorial_bump2d_komegasst PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

# `limited <k> corrected` (OF limitedSnGrad) on both paths. Asserts `limited 1` == `corrected` bit-for-bit,
# that 0.33 actually bites, and that host and device agree -- the last of which caught the device limiting
# each velocity component independently where OpenFOAM shares one limiter across the vector.
add_test(NAME limitedsngrad_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/limitedsngrad_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(limitedsngrad_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# fvOptions explicitPorositySource/DarcyForchheimer end to end. The case is BUILT here (pitzDaily plus a
# topoSet cellZone) rather than using simpleCar, because brae disagrees with OpenFOAM on simpleCar by ~50%
# with fvOptions removed from both -- see the manifest. The control is a no-porosity run.
add_test(NAME fvoptions_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/fvoptions_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(fvoptions_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# fvOptions framework + porosity, unit level: the dictionary, the cellZone selection and the resistance
# expression, asserted against OpenFOAM's source and against what the case says.
add_test(NAME fvoptions_cpp
         COMMAND test_fvoptions_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/simpleCarPorous 0)
set_tests_properties(fvoptions_cpp PROPERTIES SKIP_RETURN_CODE 77)

# limitedLinear / limitedLinearV / LUST end to end vs real OpenFOAM, each with a plain-upwind control
# run against the same oracle. The control asserts on k, not U: every one of these schemes lands within
# ~5.5e-03 of OpenFOAM on U and so does upwind, so U cannot discriminate.
add_test(NAME divschemes_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/divschemes_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(divschemes_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# linearUpwind vs real OpenFOAM on pitzDaily's own discretisation. Generates the oracle by running
# simpleFoam itself, and CONTROLS with a plain-upwind run: the deferred correction leaves the matrix
# untouched, so a port that dropped it converges and looks fine -- only the comparison can see it.
add_test(NAME linearupwind_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/linearupwind_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(linearupwind_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# kOmegaSST _cpp reference vs real OpenFOAM. The oracle is the RESIDUAL of our assembled equations at
# OpenFOAM's converged fields -- a statement about the discretisation -- with a perturbation control.
# The reference field set is generated by komegasst_vs_openfoam.sh; this test consumes it.
# limited/blended convection schemes: LUST, limitedLinear and its V variant. Asserted against exact
# properties from OpenFOAM's source (the TVD bound, both limiter limits, LUST's blend identity) with a
# control proving the V form couples the components rather than limiting them independently.
add_test(NAME limitedschemes_cpp
         COMMAND test_limitedschemes_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

# realizableKE _cpp reference vs real OpenFOAM. Same residual oracle as kOmegaSST, plus the two controls
# that say WHICH model was ported: rCmu must genuinely vary (a constant 0.09 is standard k-epsilon, which
# would still work), and S2 must use devSymm rather than the symm that kOmegaSST uses.
add_test(NAME realizablke_cpp
         COMMAND test_realizablke_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyRKE 2000)
set_tests_properties(realizablke_cpp PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME komegasst_cpp
         COMMAND test_komegasst_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailySST 2000)
set_tests_properties(komegasst_cpp PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME nonorth_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/nonorth_vs_openfoam.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/shearedChannel)
set_tests_properties(nonorth_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

add_test(NAME determinism_kepsilon
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/determinism_gate.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDaily 20)
set_tests_properties(determinism_kepsilon PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)
add_test(NAME determinism_komega
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/determinism_gate.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyKOmega 20)
set_tests_properties(determinism_komega PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# The _cpp reference for divDevReff, proven against the SAME OpenFOAM dump as the inline test above,
# with a wrong-sign check and a wall-nuEff negative control so the test is known to be able to fail.
add_executable(test_divdevreff_cpp tests/test_divdevreff_cpp.cu)
target_include_directories(test_divdevreff_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_divdevreff_cpp PRIVATE brae_core)
add_test(NAME divdevreff_cpp
         COMMAND test_divdevreff_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)

add_executable(test_divdevreff tests/test_divdevreff.cu)
target_include_directories(test_divdevreff PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_divdevreff PRIVATE brae_core)
add_test(NAME divdevreff
         COMMAND test_divdevreff ${CMAKE_CURRENT_SOURCE_DIR}/validation/kEpsCorrect)

# TURBULENT gate for the rebuilt loop: one k-epsilon SIMPLE iteration from OpenFOAM's CONVERGED solution
# must leave it there, with a laminar control proving nut really reaches the momentum equation.
add_executable(test_simple_turbulent_cpp tests/test_simple_turbulent_cpp.cu)
target_include_directories(test_simple_turbulent_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_turbulent_cpp PRIVATE brae_core)

add_executable(test_simple_sst_cpp tests/test_simple_sst_cpp.cu)
target_include_directories(test_simple_sst_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_sst_cpp PRIVATE brae_core)

add_executable(test_simple_lm_cpp tests/test_simple_lm_cpp.cu)
target_include_directories(test_simple_lm_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_lm_cpp PRIVATE brae_core)

add_executable(lm_probe tests/lm_probe.cu)
target_include_directories(lm_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(lm_probe PRIVATE brae_core)

add_executable(lm_cuda_probe tests/lm_cuda_probe.cu)
target_include_directories(lm_cuda_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(lm_cuda_probe PRIVATE brae_core)

add_executable(ami_probe tests/ami_probe.cu)
target_include_directories(ami_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ami_probe PRIVATE brae_core)

add_executable(ami_cpp_vs_device tests/ami_cpp_vs_device.cu)
target_include_directories(ami_cpp_vs_device PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ami_cpp_vs_device PRIVATE brae_core)

add_executable(ke_localize_probe tests/ke_localize_probe.cu)
target_include_directories(ke_localize_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ke_localize_probe PRIVATE brae_core)

add_executable(v2_envelope tests/v2_envelope.cu)
target_include_directories(v2_envelope PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(v2_envelope PRIVATE brae_core)

add_executable(gradu_probe tests/gradu_probe.cu)
target_include_directories(gradu_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(gradu_probe PRIVATE brae_core)

add_executable(ueqn_localize tests/ueqn_localize.cu)
target_include_directories(ueqn_localize PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(ueqn_localize PRIVATE brae_core)

add_executable(test_celllimited_cpp tests/test_celllimited_cpp.cu)
target_include_directories(test_celllimited_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_celllimited_cpp PRIVATE brae_core)

add_executable(rotordisk_probe tests/rotordisk_probe.cu)
target_include_directories(rotordisk_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(rotordisk_probe PRIVATE brae_core)

add_executable(actuationdisk_probe tests/actuationdisk_probe.cu)
target_include_directories(actuationdisk_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(actuationdisk_probe PRIVATE brae_core)
add_test(NAME celllimited_cpp
         COMMAND test_celllimited_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_simple_mrf_cpp tests/test_simple_mrf_cpp.cu)
target_include_directories(test_simple_mrf_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_mrf_cpp PRIVATE brae_core)

add_executable(test_simple_sa_cpp tests/test_simple_sa_cpp.cu)
target_include_directories(test_simple_sa_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_sa_cpp PRIVATE brae_core)

add_executable(mrf_probe tests/mrf_probe.cu)
target_include_directories(mrf_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(mrf_probe PRIVATE brae_core)
add_test(NAME simple_turbulent_cpp
         COMMAND test_simple_turbulent_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

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
# ...and brae_rhoSimpleFoam, which the registry has routed to since it was added and this list never
# named: `brae -case <compressible case>` on a fresh build execv'd a sibling that had not been built,
# so the run died at the hand-over with the case fully parsed. The comment above already said every
# registered solver belongs here; only the line was missing.
add_dependencies(brae brae_rhoSimpleFoam)

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

# brae_rhoSimpleFoam_slice: the vertical-slice port (src/applications/solvers/rhoSimpleFoam/slice).
#
# SLICE 0 is a byte-faithful COPY of the shared driver, so this target exists to prove the seam -- that the
# slice folder builds on its own, links the Tier-A infrastructure it shares rather than re-porting it, and
# reproduces brae_rhoSimpleFoam's output exactly on a real tutorial. See slice/README.md for the tier
# split and why the pimpleFoam port's defects motivate it.
add_executable(brae_rhoSimpleFoam_slice
               src/applications/solvers/rhoSimpleFoam/slice/solver/rho_simple_foam_slice.cu)
target_include_directories(brae_rhoSimpleFoam_slice PRIVATE ${BRAE_HEADER_DIRS}
                           src/applications/solvers/rhoSimpleFoam
                           src/applications/solvers/rhoSimpleFoam/slice/solver)
target_link_libraries(brae_rhoSimpleFoam_slice PRIVATE brae_core)

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
add_test(NAME ffpi_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/ffpi_vs_openfoam.sh)
set_tests_properties(ffpi_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# The other half of that boundary condition's envelope: fixedFluxPressure TOGETHER WITH MRF is refused,
# because constrainPressure.C:70 makes the flux relative and deviceConstrainPressure does not. Needs a
# fixture with a REAL rotating cellZone -- see the script's header for why pitzDaily cannot host it.
add_test(NAME v2_mrf_fixedfluxpressure_refusal
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/v2_mrf_fixedfluxpressure_refusal.sh)
set_tests_properties(v2_mrf_fixedfluxpressure_refusal PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 900)

add_test(NAME ffp_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/ffp_vs_openfoam.sh)
set_tests_properties(ffp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_executable(test_simple_adjustphi_cpp tests/test_simple_adjustphi_cpp.cu)
target_include_directories(test_simple_adjustphi_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_adjustphi_cpp PRIVATE brae_core)
add_executable(test_abl_profile tests/test_abl_profile.cu)
target_include_directories(test_abl_profile PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_abl_profile PRIVATE brae_core)
add_test(NAME abl_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/abl_vs_openfoam.sh)
set_tests_properties(abl_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME adjm_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/adjm_vs_openfoam.sh)
set_tests_properties(adjm_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_test(NAME df_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/df_vs_openfoam.sh)
set_tests_properties(df_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_test(NAME sym_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/sym_vs_openfoam.sh)
set_tests_properties(sym_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# rho.relax() on the SIMPLEC branch relaxes against the iteration's starting density (rho.prevIter()),
# both mirror arms vs real OpenFOAM mid-transient -- tests/rho_simplec_rhorelax_vs_openfoam.sh.
add_test(NAME rho_simplec_rhorelax_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_simplec_rhorelax_vs_openfoam.sh)
set_tests_properties(rho_simplec_rhorelax_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# div(phi,K)/div(phi,Ekp) is read from its OWN fvSchemes entry, not copied from the energy entry --
# both mirror arms vs real OpenFOAM in both directions, plus the by-name refusal on K alone.
add_test(NAME rho_ke_scheme_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_ke_scheme_vs_openfoam.sh)
set_tests_properties(rho_ke_scheme_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# kOmegaSST on the host mirror takes its coefficients, Prt and omega relaxation from the case, and the
# wall functions keep their own Cmu -- tests/rho_sst_coeffs_vs_openfoam.sh, at iteration 1.
add_test(NAME rho_sst_coeffs_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_sst_coeffs_vs_openfoam.sh)
set_tests_properties(rho_sst_coeffs_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# turbulence->validate() at construction, both models, both arms where the device runs the model:
# the first iteration must sit on OpenFOAM's -- tests/rho_validate_vs_openfoam.sh.
add_test(NAME rho_validate_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_validate_vs_openfoam.sh)
set_tests_properties(rho_validate_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The case's gradient limiters and laplacian limiter reach the mirror: grad(e|Ekp) cellLimited on both
# arms, `limited <psi>` on the device (refused by name on the host), `limited 0` refused on both --
# tests/rho_limiters_vs_openfoam.sh.
add_test(NAME rho_limiters_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_limiters_vs_openfoam.sh)
set_tests_properties(rho_limiters_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME tpm_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/tpm_vs_openfoam.sh)
set_tests_properties(tpm_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

add_test(NAME tf_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/validation/tf_vs_openfoam.sh)
set_tests_properties(tf_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

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

# The turbulence blow-up tripwire's DECISION, tested on the two measured sequences it has to tell apart:
# a start-up transient that crosses the ratio bar and recovers (a 5.96M-cell car case both brae and real
# OpenFOAM solve), and a run that genuinely diverges. Carries its own control -- a bar-only detector must
# stop the transient, otherwise the sequences prove nothing.
add_executable(test_turb_blowup tests/test_turb_blowup.cu)
target_include_directories(test_turb_blowup PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_turb_blowup PRIVATE brae_core)
add_test(NAME turb_blowup COMMAND test_turb_blowup)

# rhoSimpleFoam's createFields.H against real OpenFOAM. The oracle is OpenFOAM's own
# `rhoSimpleFoam -postProcess -func "writeObjects(phi,rho)"`, which builds createFields.H's field set and
# writes it without solving -- no instrumented OpenFOAM build required. Carries the control that
# distinguishes compressibleCreatePhi.H's interpolate(rho*U) from pEqn.H's interpolate(rho)*flux(U).
add_executable(test_rho_create_fields_cpp tests/test_rho_create_fields_cpp.cu)
target_include_directories(test_rho_create_fields_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_create_fields_cpp PRIVATE brae_core)
add_test(NAME rho_createfields_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_createfields_vs_openfoam.sh)

# rhoSimpleFoam's UEqn.H against OpenFOAM's own assembled momentum matrix (tools/dumpPEqn stage harness:
# rAU, internalCoeffs, boundaryCoeffs). OpenFOAM's muEff is injected so this measures the ASSEMBLY and not
# the unported compressible turbulence closure. Carries the control that distinguishes the compressible
# divDevRhoReff (dynamic mu_eff) from the incompressible divDevReff (kinematic nu_eff).
add_executable(test_rho_ueqn_cpp tests/test_rho_ueqn_cpp.cu)
target_include_directories(test_rho_ueqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_ueqn_cpp PRIVATE brae_core)
add_test(NAME rho_ueqn_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_ueqn_vs_openfoam.sh)

# rhoSimpleFoam's EEqn.H against OpenFOAM's own assembled energy equation (dumpPEqn stage harness:
# stage_Ekp, stage_he, stage_eD, stage_eSrc, stage_alphaEff). OpenFOAM's alphaEff is injected so this
# measures the ASSEMBLY. Carries the control that distinguishes the two he.name() arms --
# Ekp = 0.5|U|^2 + p/rho for `e` against K = 0.5|U|^2 for `h`.
add_executable(test_rho_eeqn_cpp tests/test_rho_eeqn_cpp.cu)
target_include_directories(test_rho_eeqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_eeqn_cpp PRIVATE brae_core)
add_test(NAME rho_eeqn_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_eeqn_vs_openfoam.sh)

# rhoSimpleFoam's pEqn.H against OpenFOAM -- BOTH branches, by running OpenFOAM twice on the same fixture
# with `transonic no` and `transonic yes` (and `consistent no`, so the driver reaches pEqn.H rather than
# pcEqn.H). The transonic branch had no stage harness at all until this gate; one was added to
# tools/dumpPEqn. Each run's control is the OTHER branch, which must not reproduce it.
add_executable(test_rho_peqn_cpp tests/test_rho_peqn_cpp.cu)
target_include_directories(test_rho_peqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_peqn_cpp PRIVATE brae_core)
add_test(NAME rho_peqn_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_peqn_vs_openfoam.sh)

# rhoSimpleFoam's pcEqn.H -- the SIMPLEC pressure equation -- against OpenFOAM, BOTH branches, with
# `consistent yes` so the driver reaches pcEqn.H. Its subsonic branch had no stage harness until this
# gate. Controls: plain SIMPLE must not reproduce the SIMPLEC matrix, and the other transonic branch
# must not either.
add_executable(test_rho_pceqn_cpp tests/test_rho_pceqn_cpp.cu)
target_include_directories(test_rho_pceqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_pceqn_cpp PRIVATE brae_core)
add_test(NAME rho_pceqn_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_pceqn_vs_openfoam.sh)

# rhoSimpleFoam END TO END: the _cpp driver runs the whole case and is compared against real OpenFOAM.
# This is the gate PORT.md requires BEFORE any .cu is written for this solver. Laminar, because the
# compressible turbulence closure is not ported -- and the binary asserts a turbulent case is REFUSED.
add_executable(test_rho_simple_step_cpp tests/test_rho_simple_step_cpp.cu)
target_include_directories(test_rho_simple_step_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_simple_step_cpp PRIVATE brae_core)
add_test(NAME rho_simple_end_to_end_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_simple_end_to_end_vs_openfoam.sh)

# The COMPRESSIBLE kEpsilon closure against OpenFOAM's own turbulence->correct(), which tools/dumpPEqn
# brackets so every input it reads and every field it writes can be compared. Carries the control that
# distinguishes the two fluxes: divU is the VOLUMETRIC one, the div operator takes the MASS one.
# The COMPRESSIBLE kOmegaSST closure against OpenFOAM's OWN kOmegaSST, instrumented. tools/dumpKOmegaSST
# is kOmegaSSTBase.C with writes added and its equations untouched, run under tools/dumpPEqn so the
# model's inputs come from the same iteration. Carries the divU flux control: the compressible reading
# takes the VOLUMETRIC flux there and the MASS flux in fvm::div, and substituting one for the other must
# be measurably worse.
add_executable(test_rho_komegasst_cpp tests/test_rho_komegasst_cpp.cu)
target_include_directories(test_rho_komegasst_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_komegasst_cpp PRIVATE brae_core)
# squareBend: OpenFOAM's OWN tutorial, UNMODIFIED, through the _cpp driver. Every other rhoSimpleFoam
# gate runs validation/sbMatched, which is derived from this case -- so this is the one that measures
# whether the port generalises past the fixture it was developed on. Compared at OpenFOAM's own
# residualControl convergence (156 iterations), not at the shipped endTime.
add_test(NAME rho_squarebend_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_squarebend_vs_openfoam.sh)
set_tests_properties(rho_squarebend_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     LABELS slow
                     TIMEOUT 1800)

add_test(NAME rho_komegasst_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_komegasst_vs_openfoam.sh)
set_tests_properties(rho_komegasst_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_komegasst_cpp
                     TIMEOUT 1800)

add_executable(test_rho_kepsilon_cpp tests/test_rho_kepsilon_cpp.cu)
target_include_directories(test_rho_kepsilon_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_rho_kepsilon_cpp PRIVATE brae_core)
# REGISTERED NOW: the closure agrees with OpenFOAM at machine precision -- epsilon 5.0e-15, k 8.9e-16,
# nut 1.7e-15, wall and interior alike -- and so does every intermediate the gate can see, because
# tools/dumpKEpsilon is OpenFOAM's own kEpsilon with writes added and its equations untouched. gradU,
# GbyNu, G, divU, both diffusivities, the mesh factors, both off-diagonal sets and both assembled systems
# each have an oracle rather than an argument. What that instrument found, in order:
#   - the closure was never given the flux, so U's inletOutlet outlet evaluated on a seeded valueFraction
#     and fvc::grad(U) read the wrong face value there (gradU exact in the interior, 1.5e-01 at the
#     boundary -- the split is what named it);
#   - k and epsilon never had updateCoeffs called on them either, so every flux-conditional patch
#     contributed nothing to its system whatever value it carried;
#   - the turbulent inlets were frozen at the case file's `value`. OpenFOAM RECOMPUTES refValue on every
#     updateCoeffs, epsilon's from k's patch values and k's from U's;
#   - the diffusion terms were assembled with `Gauss linear orthogonal` while the case asked for
#     `corrected` -- a silent scheme substitution, and the last 5e-06.
add_test(NAME rho_kepsilon_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_kepsilon_vs_openfoam.sh)
set_tests_properties(rho_simple_end_to_end_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     TIMEOUT 1800)

# div(phi,k)/div(phi,epsilon) `Gauss limitedLinear 1` end to end: sbMatched mutated to the scheme
# validation/squareBend actually ships, BOTH codes run under the mutation, so the oracle converges under
# the scheme being gated. Separate from the e2e gate because it pays its own 400-iteration OpenFOAM run.
# The fail-proof arm reverts only the scheme word and re-runs brae against the same limitedLinear
# oracle: upwind must miss the k/epsilon bound, which is also the non-vacuity statement.
add_test(NAME rho_turb_limitedlinear_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_turb_limitedlinear_vs_openfoam.sh)
set_tests_properties(rho_turb_limitedlinear_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     TIMEOUT 2400)

# The non-orthogonal corrector loop (`while (simple.correctNonOrthogonal())`), which the _cpp driver
# used to skip entirely -- one pressure solve whatever the case named. Compared at a matched EARLY
# iteration because the count is a trajectory feature: at convergence every extra pass re-solves an
# unchanged system and no converged gate can see it. The fail-proof replays the old defect (one pass
# against the 2-corrector oracle) and reads p 1.51e-01 where the arm reads 3.97e-04. Also carries the
# DEVICE arm: the CUDA driver's loop vs the OF-gated host at iteration 1 on the same mutation (its
# harness used to hardcode correctedLaplacian=false, under which the loop had nothing to correct);
# forcing the device count to 0 reads Ux 1.85e-01 against the arm's 5.7e-12.
# THE MIRROR AS A SOLVER, not as a step: `brae -case` end to end against real OpenFOAM, on BOTH arms --
# the answer, the OF-format output (real rhoSimpleFoam restarts from brae's written directory), the
# latestTime restart with an ABSOLUTE endTime, the turbulent write payload, the residualControl stop,
# and the binary-writeFormat and unknown-selector refusals. Measured on rhoBox at 200 iterations: the
# host arm p 2.95e-12 / U 2.93e-11 against OpenFOAM, the CUDA arm 3.00e-12 / 4.45e-10 against OpenFOAM
# and 2.83e-12 / 4.62e-10 against the host arm -- the two differ by their linear solvers (AMG-PCG
# against BiCGStab) and by nothing else. Every other rhoSimpleFoam gate drives the step through a
# harness that reads no controlDict and writes nothing, so none of this was gated anywhere.
# `brae -partition` + the warm reload it promises. The AMG cache serialises the agglomeration and
# re-Galerkins the values, but its LOAD path never rebuilt the per-level Galerkin gather lists -- so
# every warm run died in galDiagGatherK ("amul: an illegal memory access"). Nothing was gated on a
# warm reload, on either the compressible or the incompressible path, so nothing caught it. Asserts
# EQUALITY, not a tolerance: a cached hierarchy must reproduce a cold one exactly.
# simpleFoamV2 wrote every field's boundaryField by ECHOING the 0/ template it splices into -- so a V2
# time directory carried the start state on every patch the solve had moved (pitzDailyTurb's wall nut
# among them), and a restart from it restarted from the wrong state. Asserts the written boundary
# DIFFERS from the template (a value list alone proves nothing -- the template has one too) and that
# OpenFOAM still accepts the file.
add_test(NAME v2_written_boundary
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/v2_written_boundary.sh)
set_tests_properties(v2_written_boundary PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1200)

add_test(NAME amg_cache_warm_start
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/amg_cache_warm_start.sh)
set_tests_properties(amg_cache_warm_start PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1200)

# angledDuctExplicitFixedCoeff, OpenFOAM's own tutorial, at OpenFOAM's own convergence (639 iterations)
# -- porosity + a fixedTemperatureConstraint + a scalarFixedValueConstraint on k and epsilon. It was
# never registered, so nothing ran it; its host comparison reads p 1.9e-06, T 1.6e-06, U 1.7e-05. The
# CUDA arms check that the tutorial's TILTED SLIP wall is refused by name (mesh type `wall`, U BC
# `slip`, plane at 45 degrees) and that with an axis-independent wall the device applies all three
# fvOptions and pins the same cells as the host.
add_test(NAME rho_angledduct_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_angledduct_vs_openfoam.sh)
set_tests_properties(rho_angledduct_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     LABELS slow
                     TIMEOUT 3600)

add_test(NAME rho_mirror_solver_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_mirror_solver_vs_openfoam.sh)
set_tests_properties(rho_mirror_solver_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_relax_default_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_relax_default_vs_openfoam.sh)
set_tests_properties(rho_relax_default_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_uresidual_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_uresidual_vs_openfoam.sh)
set_tests_properties(rho_uresidual_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_tp_transient_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_tp_transient_vs_openfoam.sh)
set_tests_properties(rho_tp_transient_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_turbinlet_cmu_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_turbinlet_cmu_vs_openfoam.sh)
set_tests_properties(rho_turbinlet_cmu_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME simple_piov_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/simple_piov_vs_openfoam.sh)
set_tests_properties(simple_piov_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1800)

add_test(NAME rho_coeffdict_fallback_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_coeffdict_fallback_vs_openfoam.sh)
set_tests_properties(rho_coeffdict_fallback_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_p0_written_table_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_p0_written_table_vs_openfoam.sh)
set_tests_properties(rho_p0_written_table_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_closure_gradlimit_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_closure_gradlimit_vs_openfoam.sh)
set_tests_properties(rho_closure_gradlimit_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_p_unlimited_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_p_unlimited_vs_openfoam.sh)
set_tests_properties(rho_p_unlimited_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)
add_test(NAME rho_sbmatched_transient_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_sbmatched_transient_vs_openfoam.sh)
set_tests_properties(rho_sbmatched_transient_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae_rhoSimpleFoam
                     TIMEOUT 1800)

add_test(NAME rho_step_cuda_schemes
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_step_cuda_schemes.sh)
set_tests_properties(rho_step_cuda_schemes PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cuda
                     TIMEOUT 1800)

add_test(NAME rho_step_cuda_turbulent
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_step_cuda_turbulent.sh)
set_tests_properties(rho_step_cuda_turbulent PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cuda
                     TIMEOUT 1800)

add_test(NAME rho_nonorth_corrector_vs_openfoam
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/rho_nonorth_corrector_vs_openfoam.sh)
set_tests_properties(rho_nonorth_corrector_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_simple_step_cpp
                     TIMEOUT 900)
set_tests_properties(rho_kepsilon_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_kepsilon_cpp
                     TIMEOUT 1800)
set_tests_properties(rho_pceqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_pceqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_peqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_peqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_eeqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_eeqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_ueqn_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_ueqn_cpp
                     TIMEOUT 900)
set_tests_properties(rho_createfields_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${CMAKE_CURRENT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS test_rho_create_fields_cpp
                     TIMEOUT 900)

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
# CUDA vs the _cpp REFERENCE at stage granularity. Closes the chain OpenFOAM -> _cpp -> CUDA: every other
# GPU test compares the device against CPU code written inline in that same test, which proves consistency
# but not correctness.
add_executable(test_gpu_vs_cpp tests/test_gpu_vs_cpp.cu)
target_include_directories(test_gpu_vs_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_vs_cpp PRIVATE brae_core)
add_test(NAME gpu_vs_cpp
         COMMAND test_gpu_vs_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/matrixDumpAsym 282)
# ...and again on a TURBULENT case, where nuEff varies per cell and per boundary face. On the laminar case
# above nuEff is constant, so a kernel that mishandles a per-face diffusivity -- or reads the owner cell's
# value on a wall instead of the patch value -- would still agree perfectly.
add_test(NAME gpu_vs_cpp_turbulent
         COMMAND test_gpu_vs_cpp ${CMAKE_CURRENT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

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
