# brae's test suite and developer binaries (probes, comparison drivers, diagnostics). Added by the root
# CMakeLists.txt only with -DBRAE_BUILD_TESTS=ON (the `dev` preset): an end user's build has none of it.
#
# Executables land in the top build directory (CMAKE_RUNTIME_OUTPUT_DIRECTORY, set by the root file) and every test
# runs there, as when these registrations were part of the root file: the gate scripts find their binaries through
# BUILD=${PROJECT_BINARY_DIR}.

# Where OpenFOAM's tutorials live. $ENV{FOAM_TUTORIALS} is only set inside an OpenFOAM shell, and a
# cmake run outside one leaves it EMPTY -- which silently handed the interFoam gates the path
# "/multiphase/interFoam/..." and made every dictionary arm skip while the test still reported green.
# Defaulted to the known install exactly as tests/*.sh default OFBASHRC, and overridable.
if(DEFINED ENV{FOAM_TUTORIALS} AND NOT "$ENV{FOAM_TUTORIALS}" STREQUAL "")
    set(BRAE_OF_TUTORIALS "$ENV{FOAM_TUTORIALS}" CACHE PATH "OpenFOAM tutorials directory")
else()
    set(BRAE_OF_TUTORIALS "/usr/lib/openfoam/openfoam2412/tutorials" CACHE PATH "OpenFOAM tutorials directory")
endif()
if(NOT EXISTS "${BRAE_OF_TUTORIALS}")
    message(STATUS "OpenFOAM tutorials not found at ${BRAE_OF_TUTORIALS}; tutorial-backed arms will SKIP")
endif()

enable_testing()
find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec REQUIRED)

# The registrations, one file per subsystem (tests/cmake/). They are include()d, not add_subdirectory()d: every
# test stays in this directory, so the suite-wide properties below reach all of them.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/unit_finite_volume.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/unit_io.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/unit_linear_solvers.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/unit_turbulence.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/distributed.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/solver_simplefoam.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/solver_pimplefoam.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/solver_rhosimplefoam.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/solver_rhocentralfoam.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/solver_interfoam.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/repository.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/developer.cmake)
# What `brae -case` runs by default, without the case-solver variables below (label `default-path`)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/default_path.cmake)
# Which tests each CI tier runs (labels only; it registers nothing)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tiers.cmake)

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

# THE PRESSURE RULE'S TEST ARM. A case's GAMG on p_rgh runs brae's AMG-preconditioned PCG by default
# (device_inter_pressure_step.cu), which stops at a different residual from OpenFOAM's GAMG -- 2.3e-09 on
# RAS/DTCHull at pinned 1e-13 solves, against gates held near 1e-11. So every test runs the GAMG port
# (BRAE_PRESSURE_CASE_SOLVER=1) and keeps checking the rest of the solver to its own bound; the AMG-PCG is held
# to OpenFOAM at its measured bounds by tests/interfoam_write/amg_pcg/, which unset this. The user's decision,
# 2026-10-03.
get_property(BRAE_ALL_TESTS DIRECTORY PROPERTY TESTS)
set_property(TEST ${BRAE_ALL_TESTS} APPEND PROPERTY ENVIRONMENT "BRAE_PRESSURE_CASE_SOLVER=1")
# THE TURBULENCE RULE'S TEST ARM, the same arrangement. In interFoam a relTol-0 Gauss-Seidel smoothSolver on k,
# epsilon or omega is swept in colour order on the GPU by default (device_inter_turbulence.cu): the same smoother
# and stopping rule, another cell order, so another iterate. Every test runs the case's own order
# (BRAE_TURBULENCE_CASE_SOLVER=1) and keeps its bound; the rule is held to OpenFOAM at its measured bounds by
# tests/interfoam_write/turb_colour/, which unset this. The user's decision, 2026-10-06.
set_property(TEST ${BRAE_ALL_TESTS} APPEND PROPERTY ENVIRONMENT "BRAE_TURBULENCE_CASE_SOLVER=1")
# SIMPLEFOAM'S COLOUR RULE'S TEST ARM, the same arrangement again. The legacy simpleFoam driver sweeps a
# Gauss-Seidel smoothSolver on U and on the transported turbulence fields in colour order on the GPU by default
# (device_simple_foam.cu, decideColourRule), relTol entries included: the same smoother and stopping rule,
# another cell order, so another iterate. Every test runs OpenFOAM's order (BRAE_SIMPLE_CASE_SOLVER=1) and keeps
# its bound; the rule is held to OpenFOAM by tests/simple_colour_vs_openfoam.sh, which unsets this. The user's
# decision, 2026-10-16.
set_property(TEST ${BRAE_ALL_TESTS} APPEND PROPERTY ENVIRONMENT "BRAE_SIMPLE_CASE_SOLVER=1")
# THE DEFAULT PATH'S OWN TESTS, beside the three arrangements above, which stay as they are: the user's decision D6,
# 2026-11-20. What a customer's `brae -case` runs -- the AMG-PCG, the two colour orders -- is otherwise held only by
# the rules' dedicated gates. A test labelled `default-path` (tests/cmake/default_path.cmake) gets the three variables
# unset again: ENVIRONMENT_MODIFICATION is applied after ENVIRONMENT (CMake 3.22; the minimum here is 3.24) and
# `unset:` also removes a value inherited from ctest's own environment. Its runner, tests/default_path_run.sh, fails
# if one is still set, so a test cannot run the case-solver path under a default-path name.
foreach(_t ${BRAE_ALL_TESTS})
    get_test_property(${_t} LABELS _labels)
    if(_labels AND "default-path" IN_LIST _labels)
        set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT_MODIFICATION
                     "BRAE_PRESSURE_CASE_SOLVER=unset:" "BRAE_TURBULENCE_CASE_SOLVER=unset:" "BRAE_SIMPLE_CASE_SOLVER=unset:")
    endif()
endforeach()

# Every test above runs in the top build directory, as it did when it was defined there (add_test in a subdirectory
# would otherwise run in tests/).
get_property(_brae_tests DIRECTORY PROPERTY TESTS)
foreach(_t ${_brae_tests})
    get_test_property(${_t} WORKING_DIRECTORY _wd)
    if(NOT _wd)
        set_tests_properties(${_t} PROPERTIES WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()
endforeach()
