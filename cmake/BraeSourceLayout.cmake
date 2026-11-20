# The header search path every brae target compiles with. Included by the root CMakeLists.txt.

# OpenFOAM-style flat header resolution: .cuh co-located with .cu; include by basename.
# Collect every directory holding a header and add it to the include path.
file(GLOB_RECURSE BRAE_ALL_HEADERS ${PROJECT_SOURCE_DIR}/src/*.cuh)
set(BRAE_HEADER_DIRS "")
foreach(_h ${BRAE_ALL_HEADERS})
    get_filename_component(_d ${_h} DIRECTORY)
    list(APPEND BRAE_HEADER_DIRS ${_d})
endforeach()
list(REMOVE_DUPLICATES BRAE_HEADER_DIRS)
