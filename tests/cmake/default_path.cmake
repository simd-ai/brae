# DEFAULT-PATH TESTS, the user's decision D6 (2026-11-20): coverage of what `brae -case` runs for a customer, beside
# the suite's case-solver arrangement, which stays as it is. Every test here
#   - carries the `default-path` label, which unsets the three case-solver variables tests/CMakeLists.txt appends to
#     every test (its ENVIRONMENT_MODIFICATION block);
#   - runs through tests/default_path_run.sh, which fails if any of the three is still set and removes every other
#     BRAE_* variable brae reads, so `brae` runs as a customer's does;
#   - runs `brae`, the front door, which hands the case to the solver its controlDict names;
#   - holds the default path to a bound an EXISTING gate established, named beside it: none sets a tolerance of its
#     own. Where that gate measured the case-solver path, its bound is reused as it stands and the line says so.
# Their tier is tests/cmake/tiers.cmake's rule, measured on the GB10 (2026-11-21, ctest -j1): seven took at most 5 s
# and are in its smoke list; rotatingCylinders (8 s), shockTube (29 s) and pitzDaily (27 s: the developed-start gate
# runs an OpenFOAM precursor and OpenFOAM decomposed in two and four) are labelled slow.
#
# Included by tests/CMakeLists.txt.

set(BRAE_DEFAULT_PATH_RUN ${PROJECT_SOURCE_DIR}/tests/default_path_run.sh)

# simpleFoam: rotatingCylinders AS SHIPPED (laminar, MRF, U on a GaussSeidel smoothSolver: the colour rule's default
# order), converged, against OpenFOAM at tutorial_rotatingcylinders's bounds (solver_simplefoam.cmake), which that gate
# holds the case-solver path to. tests/tutorial_vs_openfoam.sh runs the front door as `brae .`, the same case
# directory as `-case .` (gpuSimpleFoam.cu: a bare word is the case).
add_test(NAME default_path_simplefoam_rotatingcylinders
         COMMAND ${BRAE_DEFAULT_PATH_RUN} ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 rotatingCylinders U=2e-03,p=3e-03)
set_tests_properties(default_path_simplefoam_rotatingcylinders PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800
                     LABELS "default-path;slow" ENVIRONMENT "BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# pimpleFoam: the committed pitzDaily made transient and all-Neumann, from an OpenFOAM precursor at t0 = 0.002, ten
# steps against OpenFOAM pimpleFoam at pimple_pressure_reference_developed's bound (10x OpenFOAM's own np2/np4
# spread, computed in the run; tests/pimple_pressure_reference_developed_vs_openfoam.sh), its three controls included.
# That gate runs brae_pimpleFoam itself; none of the three variables reaches pimpleFoam, so its bound is the default
# path's already: here the front door hands the case over.
add_test(NAME default_path_pimplefoam_pitzdaily
         COMMAND ${BRAE_DEFAULT_PATH_RUN} bash ${PROJECT_SOURCE_DIR}/tests/pimple_pressure_reference_developed_vs_openfoam.sh
                 $<TARGET_FILE:brae> ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/default_path_pimplefoam_work)
set_tests_properties(default_path_pimplefoam_pitzdaily PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 1800
                     LABELS "default-path;slow")

# rhoSimpleFoam: validation/rhoBox (heated duct) to steady state, its default (pre-mirror) driver against OpenFOAM at
# rho_vs_openfoam's bound (L2 1e-5 on T and p, validation/rho_vs_openfoam.sh:20).
add_test(NAME default_path_rhosimplefoam_rhobox
         COMMAND ${BRAE_DEFAULT_PATH_RUN} ${PROJECT_SOURCE_DIR}/tests/default_path_rhosimplefoam_vs_openfoam.sh
                 $<TARGET_FILE:brae>)
set_tests_properties(default_path_rhosimplefoam_rhobox PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900
                     LABELS default-path)

# interFoam: capillaryRise (W row: meshed by its Allrun, two pinned steps) on the device with the AMG-PCG on p_rgh and
# pcorr, against OpenFOAM at the pressure rule's own bound, 3.8e-12 (tests/interfoam_write/amg_pcg/capillary_rise.sh).
add_test(NAME default_path_interfoam_capillaryrise
         COMMAND ${BRAE_DEFAULT_PATH_RUN} ${PROJECT_SOURCE_DIR}/tests/default_path_interfoam_vs_openfoam.sh
                 $<TARGET_FILE:brae>)
set_tests_properties(default_path_interfoam_capillaryrise PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800
                     LABELS default-path
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR};BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# rhoCentralFoam: shockTube AS SHIPPED, through tests/rho_central_shocktube_vs_openfoam.sh unchanged, with the front
# door as its binary (BRAE_RCF_BIN, which no brae source reads): every arm of that gate at its own bounds. Its arm S is
# the customer's run; none of the three variables reaches rhoCentralFoam.
add_test(NAME default_path_rhocentralfoam_shocktube
         COMMAND ${BRAE_DEFAULT_PATH_RUN} BRAE_RCF_BIN=$<TARGET_FILE:brae>
                 ${PROJECT_SOURCE_DIR}/tests/rho_central_shocktube_vs_openfoam.sh)
set_tests_properties(default_path_rhocentralfoam_shocktube PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900
                     LABELS "default-path;slow;rhocentral" ENVIRONMENT "BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")

# THE DEFAULT simpleFoam DRIVER'S ARMS of two gates that run only the opt-in rebuilt driver (BRAE_SIMPLEFOAM_V2=1):
# write_cadence_vs_openfoam and timeloop_vs_openfoam. The arms red today for a known defect are left out, named in
# each script's header: the final directory written whatever the cadence (audit E-G13), and endTime read as an integer.
add_test(NAME default_path_write_cadence_vs_openfoam
         COMMAND ${BRAE_DEFAULT_PATH_RUN} ${PROJECT_SOURCE_DIR}/tests/default_path_write_cadence_vs_openfoam.sh
                 $<TARGET_FILE:brae>)
set_tests_properties(default_path_write_cadence_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600
                     LABELS default-path)
add_test(NAME default_path_timeloop_vs_openfoam
         COMMAND ${BRAE_DEFAULT_PATH_RUN} ${PROJECT_SOURCE_DIR}/tests/default_path_timeloop_vs_openfoam.sh
                 $<TARGET_FILE:brae>)
set_tests_properties(default_path_timeloop_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900
                     LABELS default-path)

# RUN-TO-RUN IDENTITY on the default paths that are bit-reproducible (audit E-G22): the same case twice through
# `brae -case`, every file written byte-identical, with a one-byte control (tests/default_path_determinism.sh):
# rhoSimpleFoam's default driver (rhoBox), interFoam's device loop (damBreak) and rhoCentralFoam's CUDA arm
# (shockTube). simpleFoam's is tests/determinism_gate.sh (determinism_kepsilon, determinism_komega); pimpleFoam's
# spread is measured, not asserted.
foreach(_case rhoBox damBreak shockTube)
    string(TOLOWER ${_case} _name)
    add_test(NAME default_path_determinism_${_name}
             COMMAND ${BRAE_DEFAULT_PATH_RUN} ${PROJECT_SOURCE_DIR}/tests/default_path_determinism.sh
                     $<TARGET_FILE:brae> ${_case})
    set_tests_properties(default_path_determinism_${_name} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900
                         LABELS default-path ENVIRONMENT "BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
endforeach()
