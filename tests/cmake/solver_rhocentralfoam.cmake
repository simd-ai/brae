# rhoCentralFoam (experimental). The density-based solver (`brae_rhoCentralFoam`). Labels `slow` and `rhocentral`; six
# gates also carry `known-red`, held to their registered FAILs by tools/known_red_check.py.
#
# Included by tests/CMakeLists.txt.

# The gates compare with OpenFOAM v2412 run on the same machine (they skip, 77, without it) and take minutes to two
# hours, so they carry `slow` and `rhocentral` (ctest -L rhocentral runs them all). Six of them stay red on purpose:
# their FAILs are analysed, registered open checks (tests/data/rho_central_open_checks.tsv, 24 FAIL lines in all,
# each printed with its category under the FAIL), never turned into passes and never covered by a wider bound.
# Those carry `known-red` as well, so the GB10 CI's ctest step (.github/workflows/gpu-test.yml, -LE known-red) runs the
# other seven, its next step runs these six through tools/known_red_check.py (red must be exactly the registered FAILs,
# below), and `ctest -L known-red` runs them as they are: restart 1 (round-off amplification), riemann 16 (round-off amplification
# and timestep-sequence divergence, as OpenFOAM's own decompositions diverge), nozzle 3 (round-off amplification of an
# unstable mode OpenFOAM shares; one stopping-test flip), turb 2, moving 1 and 3d 1 (an iterative solve stopping one
# sweep apart at its declared tolerance, as OpenFOAM's own runs do).
add_test(NAME rho_central_sod_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_sod_vs_openfoam.sh)
set_tests_properties(rho_central_sod_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 900
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_shocktube_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_shocktube_vs_openfoam.sh)
set_tests_properties(rho_central_shocktube_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 900
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_wedge_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_wedge_vs_openfoam.sh)
set_tests_properties(rho_central_wedge_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 2400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_forwardstep_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_forwardstep_vs_openfoam.sh)
set_tests_properties(rho_central_forwardstep_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 2400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_restart_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_restart_vs_openfoam.sh)
set_tests_properties(rho_central_restart_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral;known-red" TIMEOUT 5400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_riemann_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_riemann_vs_openfoam.sh)
set_tests_properties(rho_central_riemann_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral;known-red" TIMEOUT 5400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_viscous_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_viscous_vs_openfoam.sh)
set_tests_properties(rho_central_viscous_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 5400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_nozzle_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_nozzle_vs_openfoam.sh)
set_tests_properties(rho_central_nozzle_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral;known-red" TIMEOUT 7200
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_lts_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_lts_vs_openfoam.sh)
set_tests_properties(rho_central_lts_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 10800
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_turb_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_turb_vs_openfoam.sh)
set_tests_properties(rho_central_turb_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral;known-red" TIMEOUT 5400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_moving_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_moving_vs_openfoam.sh)
set_tests_properties(rho_central_moving_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral;known-red" TIMEOUT 5400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_3d_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_3d_vs_openfoam.sh)
set_tests_properties(rho_central_3d_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral;known-red" TIMEOUT 5400
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
# pressureInletOutletVelocity on a static mesh, CUDA and host reference arms (the host arm's update at :232)
add_test(NAME rho_central_piov_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_piov_vs_openfoam.sh)
set_tests_properties(rho_central_piov_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 1800
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
# What tells the six known-red gates' 24 registered FAILs from a regression: tools/known_red_check.py runs them and
# passes only when their FAILs are exactly the registered ones, counted per registry row and with each gate's pass
# count (tests/data/rho_central_known_red.tsv). The GB10 CI runs it as its own step; this is its no-GPU self-test,
# which the hosted build-check runs too.
add_test(NAME rho_central_known_red_check_selftest
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/known_red_check.py --selftest)
set_tests_properties(rho_central_known_red_check_selftest PROPERTIES LABELS "rhocentral" TIMEOUT 120)
add_test(NAME rho_central_refusals
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_refusals.sh)
set_tests_properties(rho_central_refusals PROPERTIES SKIP_RETURN_CODE 77 LABELS "rhocentral" TIMEOUT 600
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
# WAVE 1 / Q4 (2026-11-21): the fvSchemes and fvSolution keys as OpenFOAM v2412 names them -- the compressible
# kOmegaSST's laplacian/ddt/interpolate keys, linearUpwind's named gradient, k/omega relaxation, decayControl, any
# solver name on the diagonal solves, `linear` reconstruction, pattern keys, dotInterpolate(S,tauMC), grad(T), janaf's
# range warning, the unread-entry audit -- in two NEW tests, beside the six known-red gates (no arm added to those).
add_test(NAME rho_central_keys_refusals
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_keys_refusals.sh)
set_tests_properties(rho_central_keys_refusals PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 1200
                     RESOURCE_LOCK brae_gpu
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
add_test(NAME rho_central_keys_vs_openfoam
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/rho_central_keys_vs_openfoam.sh)
set_tests_properties(rho_central_keys_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;rhocentral" TIMEOUT 3600
                     RESOURCE_LOCK brae_gpu
                     ENVIRONMENT "BRAE_RCF_BIN=$<TARGET_FILE:brae_rhoCentralFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
