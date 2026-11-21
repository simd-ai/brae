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
# system with neither adjustPhi nor setReference. The control (BRAE_NO_PREF=1) diverges at step two.
add_test(NAME pimple_pressure_reference
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_pressure_reference_vs_openfoam.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/pimple_pressure_reference_work)
set_tests_properties(pimple_pressure_reference PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 900)

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

# Work package Q2 (pimpleFoam guard bundle). Each script's header says what it holds and how its control fails.
# The set-up refusals: configurations brae_pimpleFoam computes differently from OpenFOAM v2412 (transportModel, a missing
# turbulenceProperties, deltaT or endTime, named ddt entries, moving-mesh restart / cyclic / wall distance / forces /
# update cadence / outer-corrector motion, function objects under adjustTimeStep, the LES and DES forms brae does not
# reproduce), each refused by name under `brae check --setup`, and the nearest supported form of each set up. Small boxes
# written by tests/pimple_box_case.py; no OpenFOAM needed. About 80 s on the GB10 (two dozen set-ups on the GPU).
add_test(NAME pimple_setup_refusals
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_setup_refusals.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_BINARY_DIR}/pimple_setup_refusals_work)
set_tests_properties(pimple_setup_refusals PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow" TIMEOUT 900)

# The write grid after a latestTime restart under writeControl runTime: brae's written times against OpenFOAM's, restarting
# from its own write, from OpenFOAM's, and OpenFOAM from brae's; control BRAE_CONTROL_WRITE_GRID_ABSOLUTE=1. About 60 s.
add_test(NAME pimple_restart_write_grid_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_restart_write_grid_vs_openfoam.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/pimple_restart_write_grid_work)
set_tests_properties(pimple_restart_write_grid_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow" TIMEOUT 900)

# The turbulence fields' health on the transient loop: a natural divergence whose epsilon solve goes NaN while p and U are
# still finite, injected NaN/Inf in k, epsilon and omega, a healthy run, and the BRAE_ALLOW_NONFINITE=1 opt-out as control;
# OpenFOAM's own run of the divergence reported beside it when present. About 90 s.
add_test(NAME pimple_turbulence_health
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_turbulence_health.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/pimple_turbulence_health_work)
set_tests_properties(pimple_turbulence_health PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow" TIMEOUT 900)

# brae's IDDES hmax (the cell's bounding box) against OpenFOAM's maxDeltaxyz and maxDeltaxyzCubeRoot fields, written by
# pimpleFoam -postProcess: they agree on a Cartesian box, where brae sets up, and differ on a sheared one, where brae refuses
# and reports the difference this gate measures. Control BRAE_CONTROL_LES_MESH_UNCHECKED=1. About 20 s.
add_test(NAME pimple_iddes_hmax_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_iddes_hmax_vs_openfoam.sh
                 $<TARGET_FILE:brae_pimpleFoam>
                 ${PROJECT_BINARY_DIR}/pimple_iddes_hmax_work)
set_tests_properties(pimple_iddes_hmax_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow" TIMEOUT 900)

# adjustTimeStep on OpenFOAM's clock (M4, issue #53): pitzDaily step by step against OpenFOAM's own log -- every deltaT
# replayed through OpenFOAM's rules to the bit, the write times, the fields at each write, four restarts, two controls.
add_test(NAME pimple_adjusttimestep_vs_openfoam COMMAND bash ${PROJECT_SOURCE_DIR}/tests/pimple_adjusttimestep_vs_openfoam.sh)
set_tests_properties(pimple_adjusttimestep_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800
                     ENVIRONMENT "BRAE_BIN=$<TARGET_FILE:brae_pimpleFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
