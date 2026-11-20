# What `cmake --install` puts in bin/. Included by the root CMakeLists.txt after src/.

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
# Every solver `brae` dispatches to is in the set: an installed `brae` handed a compressible, a VOF or a
# density-based case execs brae_rhoSimpleFoam, brae_interFoam or brae_rhoCentralFoam beside it, which the set once
# left out (install.sh copies the same five).
foreach(exe IN ITEMS brae brae-agent brae_pimpleFoam brae_rhoSimpleFoam brae_interFoam brae_rhoCentralFoam)
    if(TARGET ${exe})
        set_target_properties(${exe} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
        install(TARGETS ${exe} RUNTIME DESTINATION bin)
    endif()
endforeach()
