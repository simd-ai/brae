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

# Shared settings and dependencies (cmake/), before any target exists
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(BraeCompilerOptions)
include(BraeDependencies)
include(BraeSourceLayout)

# ---------------------------------------------------------------------------
# What a build makes. An end user's build is the solvers and the node agent; the test suite (hundreds of gate and
# unit executables) and the developer binaries are built only when asked for: -DBRAE_BUILD_TESTS=ON, or the `dev`
# preset in CMakePresets.json.
# ---------------------------------------------------------------------------
option(BRAE_BUILD_TESTS "Build the test suite and the developer binaries (probes, comparison drivers)" OFF)

# Every executable and the static library land in the top build directory, whichever directory defines them: the gate
# scripts, install.sh and the release packaging find them there (BUILD=<build>/brae, <build>/libbrae_core.a).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_subdirectory(src)
include(BraeInstall)

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
