# The solver executables. `brae` is the front door: it runs steady incompressible cases itself and hands every
# other case to the solver binary beside it (solvers/common/solver_dispatch.cuh), so building `brae` builds them
# all.

# The single-GPU device-resident solver app: brae -case <caseDir>
add_executable(brae simpleFoam/gpuSimpleFoam.cu)
target_include_directories(brae PRIVATE ${BRAE_HEADER_DIRS} pimpleFoam)
target_link_libraries(brae PRIVATE brae_core)
# `brae --version`: the release, the commit the build was configured at (a later commit without a reconfigure keeps the older
# one, so the line says "configured at"), and the CUDA architectures the device code was compiled for
execute_process(COMMAND git -C ${PROJECT_SOURCE_DIR} rev-parse --short=12 HEAD
                OUTPUT_VARIABLE BRAE_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git -C ${PROJECT_SOURCE_DIR} status --porcelain --untracked-files=no
                OUTPUT_VARIABLE BRAE_GIT_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
# A source tarball has no .git, and a delivered binary must still name its source: whoever packs the tree from a commit
# writes that commit into BRAE_SOURCE_COMMIT at its root (one line, a hex hash), and --version says it was stamped there.
# Without either, the commit is "unknown" and --version says why.
set(BRAE_COMMIT_ORIGIN "git")
if(NOT BRAE_GIT_COMMIT AND EXISTS ${PROJECT_SOURCE_DIR}/BRAE_SOURCE_COMMIT)
    file(STRINGS ${PROJECT_SOURCE_DIR}/BRAE_SOURCE_COMMIT _brae_stamp LIMIT_COUNT 1)
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
add_executable(brae_pimpleFoam pimpleFoam/gpuPimpleFoam.cu)
target_include_directories(brae_pimpleFoam PRIVATE ${BRAE_HEADER_DIRS} simpleFoam)
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
# ...and brae_rhoCentralFoam, with its registry row.
add_dependencies(brae brae_rhoCentralFoam)

# brae_interFoam: transient two-phase VoF. Gated against real OpenFOAM on damBreak (alpha 3.4e-09)
# and capillaryRise (0.7%, the case where surface tension IS the answer).
add_executable(brae_interFoam interFoam/braeInterFoam.cu)
target_include_directories(brae_interFoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_interFoam PRIVATE brae_core)

# brae_rhoSimpleFoam: steady compressible solver (subsonic, laminar)
add_executable(brae_rhoSimpleFoam rhoSimpleFoam/gpuRhoSimpleFoam.cu)
target_include_directories(brae_rhoSimpleFoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_rhoSimpleFoam PRIVATE brae_core)

# brae_rhoCentralFoam: OpenFOAM v2412's rhoCentralFoam (density-based, central-upwind, Kurganov/Tadmor), EXPERIMENTAL,
# single GPU. Scope, each step closed by a gate against OpenFOAM v2412 (ARCHITECTURE.md sec. 6): inviscid, laminar and
# kOmegaSST; hePsiThermo with hConst or janaf, const or sutherland; a fixed deltaT, adjustTimeStep or LTS; a static mesh
# or dynamicMotionSolverFvMesh with velocityComponentLaplacian; restart; the boundary conditions of the seven v2412
# tutorials (rhoCentralControls.cuh refuses the rest by name). The host reference (rhoCentralStep_cpp.cu) and the CUDA
# arm (rhoCentralStep.cu) share the case, the loop and the writer (rhoCentralFoam.cu).
add_executable(brae_rhoCentralFoam
               rhoCentralFoam/gpuRhoCentralFoam.cu
               rhoCentralFoam/rhoCentralFoam.cu
               rhoCentralFoam/rhoCentralControls.cu
               rhoCentralFoam/rhoCentralCreateFields_cpp.cu
               rhoCentralFoam/rhoCentralFlux_cpp.cu
               rhoCentralFoam/rhoCentralStep_cpp.cu
               rhoCentralFoam/rhoCentralFlux.cu
               rhoCentralFoam/rhoCentralStep.cu
               rhoCentralFoam/rhoCentralViscous.cu
               rhoCentralFoam/rhoCentralSlipBCs.cu)
target_include_directories(brae_rhoCentralFoam PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(brae_rhoCentralFoam PRIVATE brae_core)
