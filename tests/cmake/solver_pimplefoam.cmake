# pimpleFoam. The transient incompressible solver (`brae_pimpleFoam`) and its PIMPLE loop contracts.
#
# Included by tests/CMakeLists.txt.

add_executable(test_courant_device ${PROJECT_SOURCE_DIR}/tests/test_courant_device.cu)
target_include_directories(test_courant_device PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_courant_device PRIVATE brae_core)
add_test(NAME courant_device COMMAND test_courant_device)

add_executable(test_motion_contract ${PROJECT_SOURCE_DIR}/tests/test_motion_contract.cu)
target_include_directories(test_motion_contract PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_motion_contract PRIVATE brae_core)
add_test(NAME motion_contract COMMAND test_motion_contract)

add_executable(test_ddt_corr_backward ${PROJECT_SOURCE_DIR}/tests/test_ddt_corr_backward.cu)
target_include_directories(test_ddt_corr_backward PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_ddt_corr_backward PRIVATE brae_core)
add_test(NAME ddt_corr_backward COMMAND test_ddt_corr_backward)

add_executable(test_pimple_loop_contract ${PROJECT_SOURCE_DIR}/tests/test_pimple_loop_contract.cu)
target_include_directories(test_pimple_loop_contract PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_pimple_loop_contract PRIVATE brae_core)
add_test(NAME pimple_loop_contract COMMAND test_pimple_loop_contract)

add_executable(test_correct_uf ${PROJECT_SOURCE_DIR}/tests/test_correct_uf.cu)
target_include_directories(test_correct_uf PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_correct_uf PRIVATE brae_core)
add_test(NAME correct_uf COMMAND test_correct_uf)

add_executable(test_ddt_corr ${PROJECT_SOURCE_DIR}/tests/test_ddt_corr.cu)
target_include_directories(test_ddt_corr PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ddt_corr PRIVATE brae_core)
add_test(NAME ddt_corr COMMAND test_ddt_corr)

add_executable(test_mesh_flux_relative ${PROJECT_SOURCE_DIR}/tests/test_mesh_flux_relative.cu)
target_include_directories(test_mesh_flux_relative PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_mesh_flux_relative PRIVATE brae_core)
add_test(NAME mesh_flux_relative COMMAND test_mesh_flux_relative)
set_tests_properties(mesh_flux_relative PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

# PIMPLE foundation: transient pimpleStep() steady-limit consistency vs step() + transient stability (skips if the
# case fixture is absent, like ami_oracle -- validation/ is local-only on CI).
add_executable(test_gpu_pimple ${PROJECT_SOURCE_DIR}/tests/test_gpu_pimple.cu)
target_include_directories(test_gpu_pimple PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_pimple PRIVATE brae_core)
add_test(NAME gpu_pimple COMMAND test_gpu_pimple ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(gpu_pimple PROPERTIES SKIP_RETURN_CODE 125)

# Driver-level ctest: run the brae_pimpleFoam EXECUTABLE end-to-end on a transient variant of the committed pitzDaily
# (kEpsilon URANS) -- exercises the controlDict/fvSchemes(ddt)/fvSolution(PIMPLE) parse + turbulence I/O + time loop +
# OF writer. Fixture is committed, so this runs on CI (skips only if it is somehow absent).
add_test(NAME pimple_run
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_run.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/pimple_run_work)
set_tests_properties(pimple_run PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 300)

# THE PRESSURE REFERENCE on the transient driver, against real OpenFOAM pimpleFoam: pitzDaily staged
# all-Neumann (its outlet p and U zeroGradient), where OpenFOAM's p.needReference() is true. The driver
# never set DeviceSimpleControls::needRef, so the shared solver took the default and ran the singular
# system with neither adjustPhi nor setReference.
#
# The gate: the developed start. An OpenFOAM precursor runs to t0 = 0.002, every equation is then solved to 1e-12,
# and brae is bounded by 10x OpenFOAM's own np2/np4 spread, computed in the run. Controls: no reference
# (BRAE_NO_PREF=1, diverges at the first step), a wrong limiter gradient and a wrong limiter (BRAE_CONTROL_LIMITER),
# each red by > 100x. In the smoke tier although it takes about 22 s (tests/cmake/tiers.cmake says why).
add_test(NAME pimple_pressure_reference_developed
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_pressure_reference_developed_vs_openfoam.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/pimple_pressure_reference_developed_work)
set_tests_properties(pimple_pressure_reference_developed PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 1800)
# DIAGNOSTIC, not a gate: the same reference from the UNIFORM start. There k[N] - k[P] = 0 on every face, so
# limitedLinear is decided by the rounding sign of grad(k); OpenFOAM's own serial and 2-rank runs choose differently on
# 49% of the faces, and the bounds (U 1e-3, p 5e-3, one GB10 measurement) follow the compiler and the GPU target.
# Kept, with its bounds as they were, because it still shows where two builds round differently. Labelled
# `diagnostic`: out of the smoke and full selections, run by gpu-full.yml in a step that never fails the job, and by
# hand with `ctest --preset diagnostic`. Its control (BRAE_NO_PREF=1) diverges at step two.
add_test(NAME pimple_pressure_reference
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_pressure_reference_vs_openfoam.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/pimple_pressure_reference_work)
set_tests_properties(pimple_pressure_reference PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 900 LABELS diagnostic)

# PIMPLE/momentumPredictor: the same case with the switch on and off must give DIFFERENT fields (and the
# default must equal `on`). See the header of the script for why the log line is not enough.
add_test(NAME momentum_predictor
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/momentum_predictor.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/momentum_predictor_work)
set_tests_properties(momentum_predictor PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 300)

# The same transient case run through `brae` itself: the case says `application pimpleFoam`, so the solver
# registry (solvers/common/solver_dispatch.cuh) must hand it to brae_pimpleFoam and produce the identical
# end-to-end result. Guards the single-command CLI: `brae` on a transient case must not fall through to SIMPLE.
add_test(NAME pimple_dispatch
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_run.sh
                 $<TARGET_FILE:brae>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/pimple_dispatch_work)
set_tests_properties(pimple_dispatch PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 300)

# The transient solver's refusals: MRF / fvOptions / adjustTimeStep are applied by the STEADY driver and not yet
# by this one, so a case using them must stop rather than be marched with the physics quietly missing. Includes
# the negative controls (an INACTIVE MRF zone, and a clean case) so the guards cannot creep. Dict-only, no GPU.
add_test(NAME pimple_guards
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_guards.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_BINARY_DIR}/pimple_guards_work)
set_tests_properties(pimple_guards PROPERTIES TIMEOUT 120)

# adjustTimeStep on OpenFOAM's clock (issue #53): pitzDaily step by step against OpenFOAM's own log -- every deltaT
# replayed through OpenFOAM's rules to the bit, the write times, the fields at each write, four restarts, two controls.
add_test(NAME pimple_adjusttimestep_vs_openfoam COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_adjusttimestep_vs_openfoam.sh)
set_tests_properties(pimple_adjusttimestep_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800
                     ENVIRONMENT "BRAE_BIN=$<TARGET_FILE:brae_pimpleFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# The flux-dependent boundary updates at OpenFOAM's points within a step: total pressure at every pressure assembly,
# pressureInletOutletVelocity after every corrector, the k and epsilon inlets and switches in the turbulence correction.
# RAS/TJunction with tight solves, every entry of every field at steps 1-3 against OpenFOAM, four controls.
add_test(NAME pimple_tjunction_boundary_vs_openfoam COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_tjunction_boundary_vs_openfoam.sh)
set_tests_properties(pimple_tjunction_boundary_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800
                     ENVIRONMENT "BRAE_BIN=$<TARGET_FILE:brae_pimpleFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
