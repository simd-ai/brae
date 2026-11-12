cmake_minimum_required(VERSION 3.24)

# Target NVIDIA GB10 / Grace Blackwell (sm_121). Override with
# -DCMAKE_CUDA_ARCHITECTURES=90 etc. on older toolkits. Phase 1 is host code in .cu
# files (compiled by nvcc); the arch matters once kernels land in the G-phases.
set(BRAE_DEFAULT_CUDA_ARCH 121)
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES ${BRAE_DEFAULT_CUDA_ARCH}
        CACHE STRING "CUDA architectures cf is built for" FORCE)
endif()

project(brae VERSION 0.4.1 LANGUAGES CXX CUDA)

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
    src/functionObjects/forces_device.cu
    src/parallel/pstream/cf_pstream.cu
    src/parallel/pstream/device_halo.cu
    src/parallel/pstream/device_reduce.cu
    src/OpenFOAM/meshes/interface/processor_interface.cu
    src/OpenFOAM/db/foam_token_reader.cu
    src/OpenFOAM/db/dynamicLibrary/codedBase/coded_library.cu
    src/OpenFOAM/primitives/functions/Function1/Coded/codedFunction1.cu
    src/meshTools/PatchFunction1/CodedField/codedPatchFunction1.cu
    src/finiteVolume/expressions/PatchFunction1/patchExprFunction1.cu
    src/OpenFOAM/meshes/primitive_mesh.cu
    src/finiteVolume/fvMesh/fv_geometry.cu
    src/finiteVolume/fvMesh/device_fv_geometry.cu
    src/finiteVolume/fvMesh/fv_patch.cu
    src/finiteVolume/fvMesh/fvPatches/constraint/cyclicACMI/cyclic_acmi_cpp.cu
    src/OpenFOAM/decompose/scotch_decomposition.cu
    src/OpenFOAM/decompose/domain_decomposition.cu
    src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nut_wall_function.cu
    src/finiteVolume/finiteVolume/fvc.cu
    src/finiteVolume/finiteVolume/fvm.cu
    src/OpenFOAM/matrices/ldu_spmv.cu
    src/OpenFOAM/matrices/pcg.cu
    src/OpenFOAM/matrices/gamg.cu
    src/OpenFOAM/matrices/pbicgstab.cu
    src/OpenFOAM/matrices/pbicg.cu
    src/meshTools/AMIInterpolation/faceAreaWeightAMI/face_area_weight_ami_cpp.cu
    src/finiteVolume/fvMesh/fvPatches/constraint/cyclicAMI/cyclic_ami_cpp.cu
    src/OpenFOAM/matrices/smooth_solver_cpp.cu
    src/matrices/lduMatrix/solvers/GAMG/GAMGAgglomerations/pair_gamg_agglomeration_cpp.cu
    src/matrices/lduMatrix/solvers/GAMG/gamg_solver_cpp.cu
    src/meshTools/solidBodyMotionFunctions/solid_body_motion_function_cpp.cu
    src/OpenFOAM/meshes/meshShapes/face/face_cpp.cu
    src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/primitive_patch_cpp.cu
    src/meshTools/cellDist/patchWave/patch_wave_cpp.cu
    src/meshTools/cellDist/patchWave/device_patch_wave.cu
    src/OpenFOAM/meshes/primitiveMesh/edges/mesh_edges_cpp.cu
    src/OpenFOAM/meshes/polyMesh/mapPolyMesh/map_poly_mesh_cpp.cu
    src/OpenFOAM/meshes/polyMesh/polyTopoChange/poly_topo_change_cpp.cu
    src/OpenFOAM/meshes/primitiveMesh/cellCells/mesh_cell_cells_cpp.cu
    src/dynamicMesh/polyTopoChange/removeFaces/remove_faces_cpp.cu
    src/dynamicMesh/polyTopoChange/hexRef8/hex_ref8_cpp.cu
    src/dynamicFvMesh/dynamicRefineFvMesh/dynamic_refine_fv_mesh_cpp.cu
    src/applications/solvers/interFoam/inter_amr_cpp.cu
    src/dynamicMesh/pointPatchDist/point_patch_dist_cpp.cu
    src/rigidBodyDynamics/rigid_body_motion_cpp.cu
    src/rigidBodyMeshMotion/rigid_body_mesh_motion_cpp.cu
    src/meshTools/twoDPointCorrector/two_d_point_corrector_cpp.cu
    src/finiteVolume/interpolation/volPointInterpolation/vol_point_interpolation_cpp.cu
    src/finiteVolume/interpolation/volPointInterpolation/point_constraints_cpp.cu
    src/waveModels/derivedPointPatchFields/waveMaker/wave_maker_point_patch_vector_field_cpp.cu
    src/fvMotionSolver/fvMotionSolvers/displacement/laplacian/displacement_laplacian_fv_motion_solver_cpp.cu
    src/fvMotionSolver/fvMotionSolvers/displacement/laplacian/device_displacement_laplacian_assembly.cu
    src/dynamicFvMesh/dynamicMotionSolverFvMesh/dynamic_motion_solver_fv_mesh_cpp.cu
    src/dynamicFvMesh/dynamicMotionSolverFvMesh/device_swept_volumes.cu
    src/matrices/lduMatrix/lduMatrix/blas1.cu
    src/matrices/lduMatrix/lduMatrix/reductions.cu
    src/matrices/lduMatrix/lduMatrix/device_spmv.cu
    src/matrices/lduMatrix/preconditioners/DILUPreconditioner/device_dilu.cu
    src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress_cpp.cu
    src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonian_cpp.cu
    src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonian.cu
    src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/realizableKE/realizableKE_cpp.cu
    src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu
    src/TurbulenceModels/turbulenceModels/LES/LESdeltas/les_delta_cpp.cu
    src/TurbulenceModels/turbulenceModels/LES/kEqn/les_kEqn_cpp.cu
    src/TurbulenceModels/turbulenceModels/LES/kEqn/device_les_keqn.cu
    src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu
    src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras_cpp.cu
    src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cu
    src/finiteVolume/cfdTools/general/fvOptions/rotorDiskSource_cpp.cu
    src/finiteVolume/cfdTools/general/fvOptions/actuationDiskSource_cpp.cu
    src/finiteVolume/cfdTools/general/bound/bound_cpp.cu
    src/finiteVolume/cfdTools/general/bound/bound_report.cu
    src/OpenFOAM/db/IOstreams/io_precision.cu
    src/finiteVolume/cfdTools/general/fvOptions/limit_temperature_report.cu
    src/TurbulenceModels/turbulenceModels/turbulenceModel/turbulence_transport.cu
    src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST.cu
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
    src/applications/solvers/interFoam/inter_driver_cpp.cu
    src/applications/solvers/interFoam/inter_writer_cpp.cu
    src/applications/solvers/interFoam/inter_case_cpp.cu
    src/applications/solvers/interFoam/inter_cn_restart.cu
    src/applications/solvers/interFoam/inter_turbulence_cpp.cu
    src/applications/solvers/interFoam/inter_waves_cpp.cu
    src/waveModels/waveModel/wave_model_cpp.cu
    src/waveModels/waveGenerationModels/base/wave_generation_bases_cpp.cu
    src/waveModels/waveGenerationModels/derived/StokesI/stokes_i_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/StokesII/stokes_ii_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/StokesV/stokes_v_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/cnoidal/cnoidal_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/streamFunction/stream_function_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/irregularMultiDirectional/irregular_multi_directional_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/Boussinesq/boussinesq_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/Grimshaw/grimshaw_wave_model_cpp.cu
    src/waveModels/waveGenerationModels/derived/McCowan/mccowan_wave_model_cpp.cu
    src/waveModels/waveAbsorptionModels/derived/shallowWaterAbsorption/shallow_water_absorption_cpp.cu
    src/transportModels/interfaceProperties/device_interface_properties.cu
    src/transportModels/twoPhaseMixture/device_two_phase_mixture.cu
    src/transportModels/interfaceProperties/interface_properties_cpp.cu
    src/applications/solvers/interFoam/inter_solve_cpp.cu
    src/applications/solvers/interFoam/inter_set_rdeltat_cpp.cu
    src/finiteVolume/finiteVolume/fvc/fvcSmooth/fvc_smooth_cpp.cu
    src/finiteVolume/finiteVolume/fvc/fvcSmooth/device_fvc_smooth.cu
    src/applications/solvers/interFoam/inter_peqn_cpp.cu
    src/applications/solvers/interFoam/inter_correct_phi_cpp.cu
    src/finiteVolume/fvMatrices/solvers/MULES/mules_cpp.cu
    src/finiteVolume/fvMatrices/solvers/MULES/device_mules.cu
    src/applications/solvers/interFoam/alpha_eqn_cpp.cu
    src/applications/solvers/interFoam/device_alpha_flux.cu
    src/applications/solvers/interFoam/device_alpha_step.cu
    src/applications/solvers/interFoam/device_alpha_presolve.cu
    src/applications/solvers/interFoam/device_alpha_subcycle.cu
    src/finiteVolume/finiteVolume/device_fvc_reconstruct.cu
    src/applications/solvers/interFoam/device_inter_ueqn.cu
    src/applications/solvers/interFoam/device_inter_peqn.cu
    src/applications/solvers/interFoam/device_inter_alpha_step.cu
    src/finiteVolume/cfdTools/general/device_alpha_courant.cu
    src/applications/solvers/interFoam/device_inter_pressure_step.cu
    src/applications/solvers/interFoam/device_inter_pcorr_solve.cu
    src/applications/solvers/interFoam/device_inter_step.cu
    src/applications/solvers/interFoam/device_inter_turbulence.cu
    src/applications/solvers/interFoam/inter_driver_device.cu
    src/applications/solvers/interFoam/inter_ueqn_cpp.cu
    src/finiteVolume/finiteVolume/ddtSchemes/CrankNicolsonDdtScheme/crank_nicolson_ddt_scheme_cpp.cu
    src/finiteVolume/finiteVolume/ddtSchemes/CrankNicolsonDdtScheme/device_crank_nicolson_ddt.cu
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
    src/matrices/lduMatrix/solvers/PBiCG/device_pbicg.cu
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
    src/matrices/lduMatrix/smoothers/symGaussSeidel/device_sym_gauss_seidel.cu
    src/matrices/lduMatrix/smoothers/GaussSeidel/device_colour_gauss_seidel.cu
    src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_vcycle.cu
    src/matrices/lduMatrix/solvers/PCG/device_amg_pcg.cu
    src/matrices/lduMatrix/solvers/GAMG/device_gamg_solver.cu
    src/cuda/device_cluster.cu
    src/cuda/device_cyclic.cu
    src/cuda/device_ami.cu
    src/applications/solvers/simpleFoam/device_simple_foam.cu
)
target_include_directories(brae_core PUBLIC ${BRAE_HEADER_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${SCOTCH_INCLUDE_DIR})
# Link MPI libraries directly (not the imported target) so its CXX compile flags do not
# reach nvcc, which would reject them.
target_link_libraries(brae_core PUBLIC ${MPI_CXX_LIBRARIES} ${SCOTCH_LIBRARY} ${SCOTCH_ERR_LIBRARY} ZLIB::ZLIB CUDA::nvrtc CUDA::cuda_driver)
# dlopen/dlsym for the run-time-compiled coded Function1 (codedFunction1.cu).
target_link_libraries(brae_core PUBLIC ${CMAKE_DL_LIBS})
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
# The test suite and the developer binaries are not part of the product: an end user builds and installs the
# solvers below and nothing else. Developers and CI switch them on (-DBRAE_BUILD_TESTS=ON, or the `dev` preset
# in CMakePresets.json).
# ---------------------------------------------------------------------------
option(BRAE_BUILD_TESTS "Build the test suite and the developer binaries (probes, comparison drivers)" OFF)

# The single-GPU device-resident solver app: brae -case <caseDir>
add_executable(brae src/applications/solvers/simpleFoam/gpuSimpleFoam.cu)
target_include_directories(brae PRIVATE ${BRAE_HEADER_DIRS} src/applications/solvers/pimpleFoam)
target_link_libraries(brae PRIVATE brae_core)
# `brae --version`: the release, the commit the build was configured at (a later commit without a reconfigure keeps the older
# one, so the line says "configured at"), and the CUDA architectures the device code was compiled for
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=12 HEAD
                OUTPUT_VARIABLE BRAE_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} status --porcelain --untracked-files=no
                OUTPUT_VARIABLE BRAE_GIT_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
# A source tarball has no .git, and a delivered binary must still name its source: whoever packs the tree from a commit
# writes that commit into BRAE_SOURCE_COMMIT at its root (one line, a hex hash), and --version says it was stamped there.
# Without either, the commit is "unknown" and --version says why.
set(BRAE_COMMIT_ORIGIN "git")
if(NOT BRAE_GIT_COMMIT AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/BRAE_SOURCE_COMMIT)
    file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/BRAE_SOURCE_COMMIT _brae_stamp LIMIT_COUNT 1)
    string(STRIP "${_brae_stamp}" _brae_stamp)
    if(_brae_stamp MATCHES "^[0-9a-f]+$")
        string(SUBSTRING "${_brae_stamp}" 0 12 BRAE_GIT_COMMIT)
        set(BRAE_COMMIT_ORIGIN "stamp")
    else()
        message(FATAL_ERROR "BRAE_SOURCE_COMMIT holds '${_brae_stamp}', not a commit hash")
    endif()
endif()
if(NOT BRAE_GIT_COMMIT)
    set(BRAE_GIT_COMMIT "unknown")
    set(BRAE_COMMIT_ORIGIN "none")
elseif(BRAE_GIT_DIRTY)
    string(APPEND BRAE_GIT_COMMIT "+modified")
endif()
# A list of architectures ("80;90", as CI builds) would split the definition at the ';' into two broken ones
# (`/bin/sh: Unterminated quoted string`), so --version gets them comma-separated.
string(REPLACE ";" "," BRAE_CUDA_ARCHS_LIST "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_definitions(brae PRIVATE BRAE_VERSION_STRING="${PROJECT_VERSION}" BRAE_GIT_COMMIT="${BRAE_GIT_COMMIT}"
                           BRAE_COMMIT_ORIGIN="${BRAE_COMMIT_ORIGIN}" BRAE_CUDA_ARCHS_STRING="${BRAE_CUDA_ARCHS_LIST}")

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
# ...and brae_interFoam, added to this list at the same time as its registry row, because the comment
# above says every registered solver belongs here and the last one to be added was not.
add_dependencies(brae brae_interFoam)

# brae_interFoam: transient two-phase VoF. Gated against real OpenFOAM on damBreak (alpha 3.4e-09)
# and capillaryRise (0.7%, the case where surface tension IS the answer).
add_executable(brae_interFoam src/applications/solvers/interFoam/braeInterFoam.cu)
target_include_directories(brae_interFoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_interFoam PRIVATE brae_core)

# brae_rhoSimpleFoam: steady compressible solver (subsonic, laminar)
add_executable(brae_rhoSimpleFoam src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu)
target_include_directories(brae_rhoSimpleFoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_rhoSimpleFoam PRIVATE brae_core)

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
# Every solver `brae` dispatches to is in the set: an installed `brae` handed a compressible or a VOF case execs
# brae_rhoSimpleFoam or brae_interFoam beside it, which the set once left out.
foreach(exe IN ITEMS brae brae-agent brae_pimpleFoam brae_rhoSimpleFoam brae_interFoam)
    if(TARGET ${exe})
        set_target_properties(${exe} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
        install(TARGETS ${exe} RUNTIME DESTINATION bin)
    endif()
endforeach()

# ---- tests ------------------------------------------------------------------------------------------------------
#
# With the suite off, `ctest` must not report an empty run as a pass: a gate script that builds without
# -DBRAE_BUILD_TESTS=ON would otherwise go green on zero tests. The one test it then finds fails and says why.
enable_testing()
if(BRAE_BUILD_TESTS)
    add_subdirectory(tests)
else()
    add_test(NAME brae_tests_not_built COMMAND ${CMAKE_COMMAND} -E false)
    message(STATUS "brae: the test suite is not built (configure with -DBRAE_BUILD_TESTS=ON, or --preset dev)")
endif()
