# Language standards, the CUDA flags every translation unit is built with, and the default build type.
# Included by the root CMakeLists.txt before any target exists.

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
