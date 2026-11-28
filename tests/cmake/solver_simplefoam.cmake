# simpleFoam. The steady incompressible solver (`brae`): equation assemblies, the device step, OpenFOAM tutorials and
# the mirror gates against real OpenFOAM v2412.
#
# Included by tests/CMakeLists.txt.

add_executable(test_forces ${PROJECT_SOURCE_DIR}/tests/test_forces.cu)
target_include_directories(test_forces PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_forces PRIVATE brae_core)
add_test(NAME forces
         COMMAND test_forces ${PROJECT_SOURCE_DIR}/validation/pitzDaily282 282)

# The _cpp reference for UEqn.H, validated by decomposition against the same OpenFOAM momentum dump:
# core vs OpenFOAM, then the dev2 term's effect isolated, then relaxation, then the MRF/fvOptions refusals.
# The CUDA UEqn against the _cpp reference, field by field (decomposed: diag/upper/lower, the three
# sources, and every boundary coefficient on every patch). Laminar and turbulent, because with constant
# nuEff a per-face-diffusivity error cannot show up.
add_executable(test_ueqn_cuda ${PROJECT_SOURCE_DIR}/tests/test_ueqn_cuda.cu)
target_include_directories(test_ueqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ueqn_cuda PRIVATE brae_core)
add_test(NAME ueqn_cuda
         COMMAND test_ueqn_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME ueqn_cuda_turbulent
         COMMAND test_ueqn_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

# One binary, one scheme each: CUDA against the _cpp reference at MATRIX granularity for every ported
# div(phi,U) scheme. The tolerances differ by scheme on purpose -- the r-ratio limiters divide by a face
# difference that approaches zero and land at 1e-12/1e-13, while linearUpwindV and LUST hit 1e-16.
foreach(_sch limitedLinear limitedLinearV LUST linearUpwindV)
    add_test(NAME ueqn_cuda_scheme_${_sch}
             COMMAND ${CMAKE_COMMAND} -E env BRAE_TEST_SCHEME=${_sch}
                     $<TARGET_FILE:test_ueqn_cuda> ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
endforeach()

# THE MANIFEST MUST MATCH ITS GENERATOR. tools/of_manifest.py already had a --check mode that reports
# drift, and nothing ran it -- so manifest/rhoSimpleFoam.yaml sat stale against the generator while
# still being the file PORT.md calls authoritative. It was carrying `U 1.08e-04 (gated at 2e-3)` for a
# gate whose bound is 5e-4 and which now holds eight fields. A check that exists but is registered
# nowhere is not a check; this is the registration.
# deviceAdjustPhi against OpenFOAM's own two guards. No fixture: the guards are arithmetic on three flux
# sums, so the inputs are constructed exactly -- including the cases a real mesh reaches only by accident.
add_executable(test_adjust_phi_guards ${PROJECT_SOURCE_DIR}/tests/test_adjust_phi_guards.cu)
target_include_directories(test_adjust_phi_guards PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_adjust_phi_guards PRIVATE brae_core)
add_test(NAME adjust_phi_guards COMMAND test_adjust_phi_guards)

add_executable(test_powerlaw_dict_guards ${PROJECT_SOURCE_DIR}/tests/test_powerlaw_dict_guards.cu)
target_include_directories(test_powerlaw_dict_guards PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_powerlaw_dict_guards PRIVATE brae_core)
add_test(NAME powerlaw_dict_guards COMMAND test_powerlaw_dict_guards)

add_executable(test_frozen_bc_guard ${PROJECT_SOURCE_DIR}/tests/test_frozen_bc_guard.cu)
target_include_directories(test_frozen_bc_guard PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_frozen_bc_guard PRIVATE brae_core)
add_test(NAME frozen_bc_guard COMMAND test_frozen_bc_guard)
add_test(NAME frozen_bc_refusals COMMAND ${PROJECT_SOURCE_DIR}/tests/frozen_bc_refusals.sh)
set_tests_properties(frozen_bc_refusals PROPERTIES SKIP_RETURN_CODE 77)

# The default simpleFoam driver runs nSweeps 1 on U and the turbulence solves and symGaussSeidel on
# nuTilda whatever the case names (kept by user decision 2026-10-15) -- and must SAY so: notices on
# airFoil2D as shipped and on pitzDaily with nSweeps 2, none where the case asks what the driver runs, none
# on the rebuilt driver, which honours both entries.
add_test(NAME simple_smoother_notices COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_smoother_notices.sh)
set_tests_properties(simple_smoother_notices PROPERTIES SKIP_RETURN_CODE 77)
# The freestream family's valueFraction from U's (and p's) STORED patch values at OpenFOAM's evaluate points, on
# the legacy simpleFoam driver: laminar airFoil2D restarted from OpenFOAM's own time 3, every solve pinned, U and
# p held to round-off at 4 and 6; the re-derivation it replaced (BRAE_CONTROL_FREESTREAM_REEVALUATE=1) fails.
add_test(NAME simple_freestream_stored_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_freestream_stored_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(simple_freestream_stored_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# simpleFoam's boundary updates at OpenFOAM's points (totalPressure and p inletOutlet at the pressure assembly,
# pressureInletOutletVelocity after the solves, the turbulence fields' switches and inlets at their equations) and totalPressure's
# value relaxed with p: windAroundBuildings from OpenFOAM's own iteration 20, three pinned iterations, every entry held; the
# momentum assembly's points (BRAE_CONTROL_STEADY_UPDATE_POINTS_LEGACY=1) and the unrelaxed value
# (BRAE_CONTROL_TOTAL_PRESSURE_UNRELAXED=1) fail.
add_test(NAME simple_update_points_vs_openfoam COMMAND bash ${PROJECT_SOURCE_DIR}/tests/simple_update_points_vs_openfoam.sh)
set_tests_properties(simple_update_points_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800
                     ENVIRONMENT "BRAE_BIN=$<TARGET_FILE:brae>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
# simpleFoam's turbulence equations under `ddtSchemes default Euler` (squareBend as shipped): OpenFOAM's models construct
# fvm::ddt of k and epsilon whatever the solver. Three pinned iterations from rest, every entry held; the term dropped
# (BRAE_CONTROL_STEADY_TURBULENCE_DDT_IGNORED=1) fails; backward on simpleFoam and Euler with RAS on rhoSimpleFoam's default
# device path are refused by name.
add_test(NAME simple_turbulence_ddt_vs_openfoam COMMAND bash ${PROJECT_SOURCE_DIR}/tests/simple_turbulence_ddt_vs_openfoam.sh)
set_tests_properties(simple_turbulence_ddt_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800
                     ENVIRONMENT "BRAE_BIN=$<TARGET_FILE:brae>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
# omegaWallFunction's and epsilonWallFunction's G0 from the STORED wall nut in the legacy closures: pitzDailySST with Spalding walls
# at tolerance 1e-9, one iteration from OpenFOAM's own iteration 50 against real OpenFOAM; the made-again nut
# (BRAE_CONTROL_G0_NUT_RECOMPUTED=1) fails.
add_test(NAME simple_g0_stored_nut_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_g0_stored_nut_vs_openfoam.sh)
set_tests_properties(simple_g0_stored_nut_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# pimpleFoam's SIMPLEC terms on the second pressure corrector from the p entering it: pitzDaily laminar, consistent yes, one
# Euler step from OpenFOAM's developed field against real OpenFOAM pimpleFoam; the predictor's grad(p)
# (BRAE_CONTROL_SIMPLEC_PREDICTOR_GRAD=1) fails.
add_test(NAME pimple_simplec_corrector_grad_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/pimple_simplec_corrector_grad_vs_openfoam.sh)
set_tests_properties(pimple_simplec_corrector_grad_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# `Gauss LUST <grad>` takes the gradient it names, cellLimited included (scheme_parse.cuh luGradLimit): pitzDaily laminar, five
# Euler steps from OpenFOAM's developed field against real OpenFOAM pimpleFoam; LUST's correction left unlimited fails.
add_test(NAME lust_named_gradient_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/lust_named_gradient_vs_openfoam.sh)
set_tests_properties(lust_named_gradient_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# simpleFoam's colour rule (U and the turbulence fields swept in colour order on the GPU, relTol entries
# included) against OpenFOAM's converged pitzDailyTurb: the colour order converges, lands within 1e-03 of
# OpenFOAM's order and no further from OpenFOAM than it; a component swap in the fused solve fails. It unsets
# BRAE_SIMPLE_CASE_SOLVER for its own arms.
# A case's div(phi,epsilon) and div(phi,omega) naming different schemes: the model's own field takes its own entry. Control: the
# other entry setting the flags, as before.
add_test(NAME simple_second_scalar_scheme COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_second_scalar_scheme.sh)
set_tests_properties(simple_second_scalar_scheme PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
add_test(NAME simple_colour_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_colour_vs_openfoam.sh)
set_tests_properties(simple_colour_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)
# kOmegaSST's first correctNut with U's patch values as read: one iteration on a 3-D duct, both outlets, cold and restarted, against real OpenFOAM. It sets its own BRAE_SIMPLE_CASE_SOLVER.
add_test(NAME simple_sst_start_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_sst_start_vs_openfoam.sh)
set_tests_properties(simple_sst_start_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)
add_test(NAME simple_scalar_uncorrected_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_scalar_uncorrected_vs_openfoam.sh)
set_tests_properties(simple_scalar_uncorrected_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# Each transported turbulence field (k, epsilon, omega, nuTilda) stops on its OWN fvSolution entry --
# tolerance, relTol, maxIter, minIter -- in both orders, against real OpenFOAM's sweep counts on pitzDailySST,
# pitzDailyTurb and airFoil2D, each row from OpenFOAM's own iteration 100 (from rest OpenFOAM's one-ulp twin moved its
# own counts). Controls: the pair under one merged rule, the fixed 3000 cap with no minIter.
# Fail-proofed against 21d124d6: 16 of its 18 checks go red (SA's colour order already took nuTilda's own).
# It unsets BRAE_SIMPLE_CASE_SOLVER for its own arms.
add_test(NAME simple_turbulence_stop_rule_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_turbulence_stop_rule_vs_openfoam.sh)
set_tests_properties(simple_turbulence_stop_rule_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# U's sweep in OpenFOAM's order stops on the U entry's own maxIter and minIter (UFinal's on PIMPLE's final corrector),
# minIter outranking maxIter as in smoothSolver.C; it took 5000 and 0 whatever the entry said. Control:
# BRAE_CONTROL_U_FIXED_CAP=1.
add_test(NAME simple_u_stop_rule_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_u_stop_rule_vs_openfoam.sh)
set_tests_properties(simple_u_stop_rule_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# laplacianSchemes `limited corrected <psi>` on the legacy driver's momentum laplacian, every closure's (kOmegaSST, kEpsilon,
# SpalartAllmaras), the steady and the transient step, against OpenFOAM serial on snappyHexMesh's Ahmed body; `limited 0` is
# `uncorrected` byte for byte. Controls: BRAE_CONTROL_LAP_UNLIMITED=1 and =closure, each red.
add_test(NAME simple_ahmed_limited_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_ahmed_limited_vs_openfoam.sh)
set_tests_properties(simple_ahmed_limited_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 5400)
# The legacy driver's solver reading: an entry naming no `solver` is refused as OpenFOAM refuses it,
# residualControl's turbulence targets are read at the start (no false unread notice) and still gate, and the
# `Solving for U?` label names what runs (DILUPBiCGStab, the colour sweep). Fail-proofed against the binary
# before these fixes: every one of the five checks goes red, the controls do not.
add_test(NAME simple_solver_entry_checks COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_solver_entry_checks.sh)
set_tests_properties(simple_solver_entry_checks PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# The legacy simpleFoam driver writes its AMG hierarchy cache by default and reads it next run (identical
# fields); BRAE_AMG_CACHE=0 writes none.
add_test(NAME simple_amg_cache COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_amg_cache.sh)
set_tests_properties(simple_amg_cache PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# The case's grad(U)/grad(nuTilda) cellLimited entries on Spalart-Allmaras (linearUpwind's named gradient, the
# non-orthogonal correction's, Omega's grad(U)), airFoil2D with those entries, restarted from OpenFOAM's time 3:
# nuTilda to 1e-04 at t4; unlimited (BRAE_CONTROL_SA_GRAD_UNLIMITED=1, how it ran) misses by three orders.
add_test(NAME sa_gradlimit_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_gradlimit_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(sa_gradlimit_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# SpalartAllmaras with freestream boundaries on airFoil2D, simpleFoam and pimpleFoam from OpenFOAM's iteration 50: nut's
# freestream patches, and nuTilda's bound (none at construction, the patches evaluated after it); every entry at three steps
# against OpenFOAM, three controls.
add_test(NAME airfoil2d_sa_freestream_vs_openfoam COMMAND bash ${PROJECT_SOURCE_DIR}/tests/airfoil2d_sa_freestream_vs_openfoam.sh)
set_tests_properties(airfoil2d_sa_freestream_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800
                     ENVIRONMENT "BRAE_PIMPLE_BIN=$<TARGET_FILE:brae_pimpleFoam>;BRAE_OF_TUTORIALS=${BRAE_OF_TUTORIALS}")
# forceCoeffs through a simpleFoam run under the case's own controls (every step by default, writeInterval, timeStart):
# the rows at the times asked for; no entry, no file
add_test(NAME simple_force_sampling COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_force_sampling.sh)
set_tests_properties(simple_force_sampling PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# functionObjects::forces and ::forceCoeffs against OpenFOAM v2412's on the same fields (brief #17): every column, the headers,
# two controls, the lifecycle's times against OpenFOAM's own run, the restart's layout, and the refusals
add_test(NAME functionobjects_forces_vs_openfoam COMMAND ${PROJECT_SOURCE_DIR}/tests/functionobjects_forces_vs_openfoam.sh)
set_tests_properties(functionobjects_forces_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
# V2 on Spalart-Allmaras with a `calculated` nut patch (the boxy5 UAV's inlet_front): it took kEpsilon's nut
# path and stopped on the empty epsilon boundary. It must run, and match the shipped `freestream` arrangement.
add_test(NAME sa_v2_calculated_nut COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_v2_calculated_nut.sh)
set_tests_properties(sa_v2_calculated_nut PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

# Audit finding #16: the turbulence-setup guards compared boundaryField keys to patch names exactly, so
# regex/group keys (bfs2D writes every wall BC as "(upperWall|lowerWall)") disarmed them silently while
# buildField resolved the same keys correctly. The guards resolve through patchesResolvingTo now; two
# arms are discriminating (they ran silently before the fix).
add_test(NAME regex_patchkey_guards COMMAND ${PROJECT_SOURCE_DIR}/tests/regex_patchkey_guards.sh)
set_tests_properties(regex_patchkey_guards PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_v2_laplacian_parse ${PROJECT_SOURCE_DIR}/tests/test_v2_laplacian_parse.cu)
target_include_directories(test_v2_laplacian_parse PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_v2_laplacian_parse PRIVATE brae_core)
add_test(NAME v2_laplacian_parse COMMAND test_v2_laplacian_parse)

add_executable(test_ueqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_ueqn_cpp.cu)
target_include_directories(test_ueqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_ueqn_cpp PRIVATE brae_core)

# WriteControl's cadence from a controlDict: writeInterval, its older name writeFrequency, and neither; the
# same entry in the interFoam loops' cadence; timeStep with an interval below 1 refused
add_executable(test_write_control_interval ${PROJECT_SOURCE_DIR}/tests/test_write_control_interval.cu)
target_include_directories(test_write_control_interval PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_write_control_interval PRIVATE brae_core)
add_test(NAME write_control_interval COMMAND test_write_control_interval)

add_executable(test_fvoptions_cpp ${PROJECT_SOURCE_DIR}/tests/test_fvoptions_cpp.cu)
target_include_directories(test_fvoptions_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fvoptions_cpp PRIVATE brae_core)

add_executable(test_limitedschemes_cpp ${PROJECT_SOURCE_DIR}/tests/test_limitedschemes_cpp.cu)
target_include_directories(test_limitedschemes_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_limitedschemes_cpp PRIVATE brae_core)
add_test(NAME ueqn_cpp
         COMMAND test_ueqn_cpp ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_momentum_matrix ${PROJECT_SOURCE_DIR}/tests/test_momentum_matrix.cu)
target_include_directories(test_momentum_matrix PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_momentum_matrix PRIVATE brae_core)
add_test(NAME momentum_matrix
         COMMAND test_momentum_matrix ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym)

# The _cpp reference for pEqn.H, stage by stage: rAU/HbyA vs OpenFOAM's A()/H() (ops.dat), the pressure
# Laplacian vs peqn.dat, then setReference / correctFlux / relax asserted analytically, plus the refusals.
# The CUDA pEqn against the _cpp reference, stage by stage. Laminar and turbulent.
add_executable(test_peqn_cuda ${PROJECT_SOURCE_DIR}/tests/test_peqn_cuda.cu)
target_include_directories(test_peqn_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_peqn_cuda PRIVATE brae_core)
add_test(NAME peqn_cuda
         COMMAND test_peqn_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282 laminar)
add_test(NAME peqn_cuda_turbulent
         COMMAND test_peqn_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_peqn_cpp ${PROJECT_SOURCE_DIR}/tests/test_peqn_cpp.cu)
target_include_directories(test_peqn_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_peqn_cpp PRIVATE brae_core)
add_test(NAME peqn_cpp
         COMMAND test_peqn_cpp ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_peqn_matrix ${PROJECT_SOURCE_DIR}/tests/test_peqn_matrix.cu)
target_include_directories(test_peqn_matrix PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_peqn_matrix PRIVATE brae_core)
add_test(NAME peqn_matrix
         COMMAND test_peqn_matrix ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_simple_step_cuda ${PROJECT_SOURCE_DIR}/tests/test_simple_step_cuda.cu)
target_include_directories(test_simple_step_cuda PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_step_cuda PRIVATE brae_core)
add_test(NAME simple_step_cuda
         COMMAND test_simple_step_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpSimple 282 laminar)
# 10 iterations: the CUDA driver must still track the _cpp driver. Pins the AMG-hierarchy-vs-transient-
# pressure-buffer bug, which was exact at iteration 1 and wrong by 1.3e-01 at iteration 2.
add_test(NAME simple_step_cuda_loop
         COMMAND test_simple_step_cuda ${PROJECT_SOURCE_DIR}/validation/matrixDumpSimple 282 laminar 10)

add_test(NAME simple_step_cuda_turbulent
         COMMAND test_simple_step_cuda ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_simple_step_cpp ${PROJECT_SOURCE_DIR}/tests/test_simple_step_cpp.cu)
target_include_directories(test_simple_step_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_step_cpp PRIVATE brae_core)
add_test(NAME simple_step_cpp
         COMMAND test_simple_step_cpp ${PROJECT_SOURCE_DIR}/validation/matrixDumpSimple)

add_executable(test_simple_step ${PROJECT_SOURCE_DIR}/tests/test_simple_step.cu)
target_include_directories(test_simple_step PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_step PRIVATE brae_core)
add_test(NAME simple_step
         COMMAND test_simple_step ${PROJECT_SOURCE_DIR}/validation/matrixDumpSimple)

add_executable(test_simple_run ${PROJECT_SOURCE_DIR}/tests/test_simple_run.cu)
target_include_directories(test_simple_run PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_run PRIVATE brae_core)
add_test(NAME simple_run
         COMMAND test_simple_run ${PROJECT_SOURCE_DIR}/validation/matrixDumpSimple)

# DETERMINISM GATE. Two runs of the same binary on the same case must agree bit for bit. This is a
# prerequisite for the whole _cpp-vs-CUDA validation plan: a defect smaller than the run-to-run noise floor
# cannot be detected, and the floor here used to be 3.6e-02 after 20 iterations. Two cases, because the
# turbulence wall functions were one of the three scatter sites and kEpsilon/kOmegaSST take different ones.
# SKIP_RETURN_CODE 77 lets it skip cleanly when no GPU binary is present.
# DISPATCH GATE for the rebuilt simpleFoam: off changes nothing, on+supported runs, on+unsupported
# REFUSES with the reason. A path that quietly degrades is indistinguishable from a correct one.
# A regex-keyed turbulence residualControl must actually bind -- see the script for why this exists and
# for what it is not (the defect it was written to chase turned out to be a symptom of T3A diverging).
add_test(NAME simple_residualcontrol_regex
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_residualcontrol_regex.sh)
set_tests_properties(simple_residualcontrol_regex PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1800)

add_test(NAME simplefoam_v2_dispatch
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simplefoam_v2_dispatch.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(simplefoam_v2_dispatch PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# CONVERGENCE GATE: run the rebuilt path to convergence and compare with OpenFOAM's own converged fields
# (validation/pitzDailyTurb/1576, genuine simpleFoam v2412 output). The bounds are what brae's established
# solver achieves against the same reference in simple_turbulent_full. ~90 s.
add_test(NAME simplefoam_v2_convergence
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simplefoam_v2_convergence.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)
set_tests_properties(simplefoam_v2_convergence PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 600)

# NON-ORTHOGONAL CORRECTION vs real OpenFOAM on a genuinely non-orthogonal mesh, with a control requiring
# that omitting the term is clearly worse -- on a near-orthogonal mesh this gate would pass with the term
# deleted. Skips (77) where OpenFOAM is not installed.
# STOCK pitzDaily end to end: SIMPLEC + linearUpwind + bounded + corrected + kEpsilon, unmodified, vs
# real OpenFOAM. Also asserts SIMPLEC and plain SIMPLE reach the SAME fixed point -- the control that
# exposed the missing fvMatrix::faceFluxCorrection.
add_test(NAME stock_pitzdaily_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/stock_pitzdaily_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(stock_pitzdaily_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# MRF end to end on the _cpp path vs real OpenFOAM, with the control that makes it mean something: with
# the zones dropped this case has no driving force at all and every field reads 1.000 against OpenFOAM.
add_test(NAME mrf_cpp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/mrf_cpp_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/mixerVessel2D)
set_tests_properties(mrf_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one hook at a time. Its control
# removes constant/MRFProperties, which exercises the real dictionary path rather than a debug switch.
add_test(NAME mrf_cuda_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/mrf_cuda_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/mixerVessel2D)
set_tests_properties(mrf_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

# cellLimited Gauss linear <k> vs real OpenFOAM, on BOTH paths, with an unlimited control that must be
# markedly worse. This is what windAroundBuildings was refused for: linearUpwind's correction is built
# from the gradient the scheme NAMES, and it does not vanish at convergence.
add_test(NAME celllimited_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/celllimited_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/windAroundBuildingsBox)
set_tests_properties(celllimited_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# rotorDiskSource vs OpenFOAM's OWN reported drag/lift/AOA on the real 71734-cell tutorial mesh, plus a
# device-vs-_cpp check. The device implementation predated any test; this is the first thing to check it.
add_test(NAME rotordisk_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/rotordisk_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/rotorDisk)
set_tests_properties(rotordisk_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# fv::actuationDiskSource (Froude) against the Uref/a/thrust OpenFOAM writes for itself, evaluated at the
# field OpenFOAM computed that row FROM -- which makes it exact to 13 digits rather than approximate.
add_test(NAME actuationdisk_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/actuationdisk_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/turbineSiting 600)
set_tests_properties(actuationdisk_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# The SAME case through the CUDA V2 driver, end to end against real OpenFOAM at DEEP convergence. This is
# what pins the SIGN the thrust is applied with, which the .dat oracle above cannot see: it compares
# numbers computed from a given velocity, and those are identical whichever way the disk pushes. Its
# control is the same run with the turbines removed.
add_test(NAME turbinesiting_cuda_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/turbinesiting_cuda_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/turbineSiting)
set_tests_properties(turbinesiting_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

# turbulence->validate() (simpleFoam.C:92) is correctNut(), so OpenFOAM's FIRST momentum assembly runs on
# nut rebuilt from the initial k and omega and not on the case's `nut uniform 0`. Oracle: dumpSimpleFoam's
# stage_nuEff/stage_UDiag0/stage_rAU at iteration 1. Control: the file's own nut, which must disagree --
# it is the laminar first iteration this used to run.
# fvMatrix<vector>::solveSegregated skips the components polyMesh::solutionD() knocks out
# (fvMatrixSolve.C:164) and residualControl compares cmptMax over what it stored (solutionControl.C:232).
# The V2 step solved all three and reported component 0. Oracle: OpenFOAM's own log, one iteration from
# its own field; control: that fixture's Uy/Ux = 1.910, so component 0 reads 47.7% low; 3D arm proves the
# mask is derived and not hardcoded off.
# brae routes `smoothSolver` + `symGaussSeidel` to a MULTICOLOUR Gauss-Seidel sweep where
# symGaussSeidelSmoother.C sweeps in index order -- same stopping rule, less smoothing per sweep. It was
# announced nowhere and the run log said "as the case asks". Positive + two controls, no GPU and no OF.
# lduMatrix::solver::readControls reads minIter and maxIter from EACH FIELD'S own entry
# (lduMatrixSolver.C:190-208). The V2 driver read one maxIter from `p` and gave it to U as well, and
# minIter reached no solver at all. Oracle: OpenFOAM's own iteration counts under the same cap and floor.
# fvc::grad(U) resolves the gradSchemes `grad(U)` entry, and divDevReff (linearViscousStress.C:114) and
# every closure's strain and production are built from it. The V2 driver passed a literal 0.0 to the SST,
# let kEpsilon's and divDevReff's arguments default, and SA had no such parameter at all. Proximity gate:
# brae must land nearer OpenFOAM's LIMITED answer than its unlimited one, on the fields where brae is
# inside the two oracles' separation.
# endTime is an ABSOLUTE time (Time.C:785), deltaT is the step (Time.C:1067) and a time directory is
# named by the VALUE (Time.C:721-728). The V2 driver read endTime as an iteration count and never read
# deltaT. Oracle: OpenFOAM's own ordered list of time names on four shapes, one of them ratio 2.5 where a
# rounded quotient gives the wrong count.
# nSweeps (smoothSolver.C:78) is the number of smoothing sweeps between residual EVALUATIONS, and
# OpenFOAM advances its iteration count by nSweeps (:205), so every count it reports is a multiple.
# brae read the entry nowhere -- airFoil2D ships nSweeps 2 and brae reported odd counts. The negative
# fixed-sweep mode (:96-119) is refused: it deletes the convergence test and reports a residual of zero.
# linearUpwind, linearUpwindV and LUST all read the gradient name off div(phi,U)'''s own stream
# (linearUpwind.H:105, linearUpwindV.H:117-135, LUST.H). brae'''s regex demanded whitespace after
# `linearUpwind`, so V and LUST silently fell back to gradSchemes default, and it searched the whole
# divSchemes block rather than div(phi,U)'''s entry.
add_test(NAME linearupwindv_grad_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/linearupwindv_grad_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_SOURCE_DIR}/validation/windAroundBuildingsBox)
set_tests_properties(linearupwindv_grad_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1200)

add_test(NAME timeloop_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/timeloop_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/simpleBoxIO)
set_tests_properties(timeloop_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# The start directory is timeName(startTime) at the case's timeFormat and timePrecision, and startFrom defaults to
# latestTime, against real OpenFOAM on simpleFoam (the default `brae` path) and pimpleFoam; `brae check` must name the
# directory the run reads. `startTime 0.;` (as ANSA writes it) is the arm that found it.
add_test(NAME start_time_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/start_time_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_SOURCE_DIR}/validation/pimFR)
set_tests_properties(start_time_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# `#include "${FOAM_CASE}/..."`, #eval of the included variables and a coded function object's verbatim `#include`s,
# as occDrivAerStaticMesh writes them, read as OpenFOAM reads them: brae on the structured case is brae on OpenFOAM's
# own `foamDictionary -expand` of it, byte for byte
add_test(NAME case_include_eval_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/case_include_eval_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailySST)
set_tests_properties(case_include_eval_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# The turbulence model is the one simulationType names: an LES{} block under `simulationType RAS` is not read, by OpenFOAM
# or by brae (it ran the LES model, occDrivAerRotMesh's RANS stage); `simulationType LES` takes it
add_test(NAME turbulence_simulation_type_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/turbulence_simulation_type_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailySST)
set_tests_properties(turbulence_simulation_type_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# velocityDampingConstraint (occDrivAerStaticMesh's fvOption) against real OpenFOAM on one GPU: selectionMode all and a cellZone,
# pitzDaily restarted from OpenFOAM's iteration 50, 5 tight iterations; a missing UMax, selectionMode points and a missing zone
# refused by the run and by `brae check`
add_test(NAME fvoptions_velocity_damping_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/fvoptions_velocity_damping_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(fvoptions_velocity_damping_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# The V2 driver restarted from OpenFOAM's converged windAroundBuildingsBox 400/, against OpenFOAM's own restart: its first
# boundary nut from k's and epsilon's stored outlet values, as OpenFOAM's validate() reads them. Control: BRAE_IO_STORED=0.
add_test(NAME simple_v2_restart_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_v2_restart_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/windAroundBuildingsBox)
set_tests_properties(simple_v2_restart_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME turbulence_off_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/turbulence_off_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(turbulence_off_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1200)

add_test(NAME solve_report_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/solve_report_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(solve_report_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

add_test(NAME write_cadence_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/write_cadence_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(write_cadence_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME tightp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tightp_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(tightp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1500)

add_test(NAME empty_direction_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/empty_direction_vs_openfoam.sh)
set_tests_properties(empty_direction_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME legacy_symmetry_refresh
         COMMAND ${PROJECT_SOURCE_DIR}/tests/legacy_symmetry_refresh.sh)
set_tests_properties(legacy_symmetry_refresh PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

add_executable(test_case_dict_refusals ${PROJECT_SOURCE_DIR}/tests/test_case_dict_refusals.cu)
target_include_directories(test_case_dict_refusals PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_case_dict_refusals PRIVATE brae_core)
add_test(NAME case_dict_refusals COMMAND test_case_dict_refusals)

add_executable(test_continuity_from_phi ${PROJECT_SOURCE_DIR}/tests/test_continuity_from_phi.cu)
target_include_directories(test_continuity_from_phi PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_continuity_from_phi PRIVATE brae_core)
add_test(NAME mirror_continuity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/mirror_continuity.sh)
set_tests_properties(mirror_continuity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME mirror_scalar_transport_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/mirror_scalar_transport_vs_openfoam.sh)
set_tests_properties(mirror_scalar_transport_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME transonic_p_dilu
         COMMAND ${PROJECT_SOURCE_DIR}/tests/transonic_p_dilu.sh)
set_tests_properties(transonic_p_dilu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME p_limit_boundary_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/p_limit_boundary_vs_openfoam.sh)
set_tests_properties(p_limit_boundary_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME wall_coeffs_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/wall_coeffs_vs_openfoam.sh)
set_tests_properties(wall_coeffs_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME u_colour_gs_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/u_colour_gs_vs_openfoam.sh)
set_tests_properties(u_colour_gs_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

add_test(NAME precon_policy_one_rule
         COMMAND ${PROJECT_SOURCE_DIR}/tests/precon_policy_one_rule.sh)
set_tests_properties(precon_policy_one_rule PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
add_test(NAME sa_precon_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_precon_vs_openfoam.sh)
set_tests_properties(sa_precon_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_executable(test_bound_report ${PROJECT_SOURCE_DIR}/tests/test_bound_report.cu)
target_include_directories(test_bound_report PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_bound_report PRIVATE brae_core)
add_test(NAME bound_report COMMAND test_bound_report)
add_test(NAME turb_minima_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/turb_minima_vs_openfoam.sh)
set_tests_properties(turb_minima_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_test(NAME simple_bound_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_bound_vs_openfoam.sh)
set_tests_properties(simple_bound_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
add_test(NAME simple_gradu_sites_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_gradu_sites_vs_openfoam.sh)
set_tests_properties(simple_gradu_sites_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_test(NAME bound_message_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/bound_message_vs_openfoam.sh)
set_tests_properties(bound_message_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_test(NAME bound_weak_solver_stress
         COMMAND ${PROJECT_SOURCE_DIR}/tests/bound_weak_solver_stress.sh)
set_tests_properties(bound_weak_solver_stress PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_test(NAME simple_linearupwind_turbulence_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_linearupwind_turbulence_vs_openfoam.sh)
set_tests_properties(simple_linearupwind_turbulence_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
add_test(NAME motorbike_tight_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/motorbike_tight_vs_openfoam.sh)
set_tests_properties(motorbike_tight_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 3600)
# grad(k) and grad(omega) each take the case's own gradSchemes entry, named-then-default, in the legacy kOmegaSST's
# CDkOmega, laplacian correction and limitedLinear limiter: five arms (plain, limited, split, widened, default) against
# OpenFOAM's own intermediates, 2-D pitzDailySST restarted and 3-D motorBike cold. Controls: the limiter ignored
# (all, CDkOmega alone, the equations alone) and omega on k's entry, each red.
add_test(NAME simple_turb_gradlimit_2d_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_turb_gradlimit_vs_openfoam.sh 2d)
set_tests_properties(simple_turb_gradlimit_2d_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
add_test(NAME simple_turb_gradlimit_3d_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_turb_gradlimit_vs_openfoam.sh 3d)
set_tests_properties(simple_turb_gradlimit_3d_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 5400)
add_test(NAME limitedlinear_robustness_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/limitedlinear_robustness_vs_openfoam.sh)
set_tests_properties(limitedlinear_robustness_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
add_test(NAME simple_divergence_detector
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_divergence_detector.sh)
set_tests_properties(simple_divergence_detector PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
add_test(NAME bound_at_construction_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/bound_at_construction_vs_openfoam.sh)
set_tests_properties(bound_at_construction_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_test(NAME limit_temperature_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/limit_temperature_vs_openfoam.sh)
set_tests_properties(limit_temperature_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 LABELS slow)
add_test(NAME write_format_binary_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/write_format_binary_vs_openfoam.sh)
set_tests_properties(write_format_binary_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 LABELS slow)
add_test(NAME laminar_model_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/laminar_model_vs_openfoam.sh)
set_tests_properties(laminar_model_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 2400 LABELS slow)
add_test(NAME nut_wall_function_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/nut_wall_function_vs_openfoam.sh)
set_tests_properties(nut_wall_function_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 2400 LABELS slow)
add_test(NAME eeqn_limitedlinear_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/eeqn_limitedlinear_vs_openfoam.sh)
set_tests_properties(eeqn_limitedlinear_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 2400 LABELS slow)
add_test(NAME turb_limitedlinear_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/turb_limitedlinear_vs_openfoam.sh)
set_tests_properties(turb_limitedlinear_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 2400 LABELS slow)
add_test(NAME turb_precon_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/turb_precon_vs_openfoam.sh)
set_tests_properties(turb_precon_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
add_test(NAME wall_blending_refusals
         COMMAND ${PROJECT_SOURCE_DIR}/tests/wall_blending_refusals.sh)
set_tests_properties(wall_blending_refusals PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
add_test(NAME p_amg_bicgstab_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/p_amg_bicgstab_vs_openfoam.sh)
set_tests_properties(p_amg_bicgstab_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

add_test(NAME uresidual_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/uresidual_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/simpleBoxIO
                 ${PROJECT_SOURCE_DIR}/validation/cav3d_of)
set_tests_properties(uresidual_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME validate_nut_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/validate_nut_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(validate_nut_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

# The DIV SCHEME's face weight at a coupled interface, on the real pipeCyclic tutorial against real
# OpenFOAM. brae assembled every cyclic and cyclicAMI face UPWIND whatever the case named -- a defect
# invisible to ami_cpp_vs_device above, because brae's device and its reference implemented the same wrong
# scheme and agreed to 1e-16 on it. Four runs, not one: brae and OpenFOAM under BOTH the limited and the
# upwind scheme, so matching OpenFOAM is checked against a case proven to discriminate the two. Plus a
# symmetry check that needs no reference at all -- the sector's transverse pressure force must vanish, and
# the upwind interface put it at 2.2e-02 of the axial force against 2.1e-06 now.
add_test(NAME coupledinterfacescheme_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/coupledinterfacescheme_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pipeCyclic)
set_tests_properties(coupledinterfacescheme_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The SAME gate on the other blend, `bounded Gauss linear`, which at a coupled face is the patch's own
# interpolation weight -- w ~ 0.5, the furthest a weight can get from upwind's 0 or 1, so this is the widest
# of the three gaps and the one a wrong interface assembly distorts most. OpenFOAM's implicitAMI is the only
# tutorial anywhere that pairs a coupled patch with `Gauss linear` on div(phi,U); it is laminar, so its
# bounds name only U and p and the gate checks the fields the case actually wrote.
add_test(NAME linearinterfacescheme_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/coupledinterfacescheme_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/implicitAMI U=4e-02,p=8e-02)
set_tests_properties(linearinterfacescheme_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# SIMPLEC (`consistent true`) across a coupled interface, on OpenFOAM's basic/simpleFoam/implicitAMI.
# fvMatrix::H1() sums boundaryCoeffs over every coupled patch and the SIMPLEC flux term is evaluated on
# coupled boundary faces like any other; brae did neither, so on a mesh coupled only by a cyclicAMI the
# whole SIMPLEC correction stopped at the interface. Nothing in validation/ combined SIMPLEC with a
# coupled interface before this case, which is why the path was unreachable. The sharp check seeds brae
# with OpenFOAM's own converged U, p and phi: at that state the true divergence is zero, so brae's
# continuity error measures its own inconsistency directly -- 1.3e-01 before, 1.7e-03 after, with the
# momentum residual at 2.8e-07 throughout to show the UEqn was never the term at fault.
add_test(NAME simplec_interface_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simplec_interface_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/implicitAMI)
set_tests_properties(simplec_interface_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# Two stock simpleFoam tutorials that had no gate at all: they were run once, agreed, and nothing kept
# them agreeing. Both are read straight from the OpenFOAM installation, so neither needs a committed case.
# backwardFacingStep2D is the ONLY simpleFoam tutorial that names LUST, so it is the one thing standing
# behind brae's internal LUST weights. Each asserts that both codes reached a steady state before
# comparing -- squareBend reads U 2.759e-02 at its tutorial endTime and 1.773e-03 once both codes are on
# the same residual plateau, a factor of 16 from nothing but where the comparison was taken -- and each
# carries a control requiring the INITIAL field to fail the same bounds.
add_test(NAME tutorial_rotatingcylinders
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 rotatingCylinders U=2e-03,p=3e-03)
set_tests_properties(tutorial_rotatingcylinders PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# COMPARED AS A WINDOW MEAN, and the bounds below are ~2x what the mean actually achieves. This case has
# no steady state: OpenFOAM's own p residual sits at ~2.1e-02 for 20000 iterations and oscillates rather
# than falling, and against ITSELF 250 iterations apart it moves U by 1.66e-02 -- more than the 1e-02
# bound brae used to be held to at a single iterate. Averaging the last 21 writes (4000..6000) cuts
# OpenFOAM's own window-to-window variation to 6.05e-04 on U, and that mean is stationary (the window at
# 4000..6000 matches the one at 20000..29000 to 5.25e-04), which is why the window can be taken this
# early. Measured brae vs OpenFOAM: U 2.650e-03, p 3.363e-03, k 8.316e-03, omega 6.537e-04, nut 6.550e-03
# -- every one of them tighter than the single-iterate bound it replaces. See WINDOW in the script.
add_test(NAME tutorial_backwardfacingstep2d
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 backwardFacingStep2D U=6e-03,p=8e-03,k=2e-02,omega=2e-03,nut=1.5e-02
                 1e-04 6000/100 EMPTY 21)
set_tests_properties(tutorial_backwardfacingstep2d PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The MULTI-SETUP tutorials. bump2D and turbulentFlatPlate are not single cases: each ships a
# `setups.orig/` tree of one common skeleton plus a per-turbulence-model overlay, and turbulentFlatPlate
# additionally builds its mesh from a blockMeshDict.template expanded with a per-y+ grading -- so it is a
# matrix of (model x near-wall grading), not a list of cases. tests/tutorial_vs_openfoam.sh composes them
# exactly as their own Allrun does and names one as tutorial:setup[:grading].
#
# turbulentFlatPlate at y+ 1 is the tightest agreement of any tutorial gated here (U 1.9e-05, nut 3.8e-03),
# which is worth having precisely because bump2D at its own residualControl of 1e-05 is the loosest.
add_test(NAME tutorial_turbulentflatplate_komegasst
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 turbulentFlatPlate:kOmegaSST:2200 U=2e-04,p=3e-02,k=1e-03,omega=5e-04,nut=2e-02)
# its kEpsilon twin's limit: 2,174 s and 1,984 s of the 2,400 it had in two full suites (2026-10-08)
set_tests_properties(tutorial_turbulentflatplate_komegasst PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 3600)

add_test(NAME tutorial_bump2d_spalartallmaras
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 bump2D:SpalartAllmaras U=5e-03,p=5e-03,nut=1e-01,nuTilda=1e-01)
set_tests_properties(tutorial_bump2d_spalartallmaras PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

# ...and the setup brae must REFUSE. bump2D's third model is kEpsilonPhitF, which brae does not implement;
# running it as the kEpsilon it does have would be a wrong answer wearing the right label. The gate asserts
# brae exits nonzero AND names the model, so a refusal that says nothing useful fails too.
add_test(NAME tutorial_bump2d_kepsilonphitf_refused
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 bump2D:kEpsilonPhitF REFUSE:kEpsilonPhitF)
set_tests_properties(tutorial_bump2d_kepsilonphitf_refused PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# turbulentFlatPlate:kEpsilon at y+ ~ 1 -- the case that DIVERGED until the turbulence solve got the DILU
# preconditioner the case asks for. It is gated tightly on purpose: the failure it guards was not a wrong
# term but a preconditioner substitution, and the symptom of that returning is the run not finishing at
# all, so bounds near the measured 1.8e-05 on U cost nothing and say exactly what is expected.
add_test(NAME tutorial_turbulentflatplate_kepsilon
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 turbulentFlatPlate:kEpsilon:2200 U=2e-04,p=2e-03,k=3e-03,epsilon=2e-02,nut=2e-03)
set_tests_properties(tutorial_turbulentflatplate_kepsilon PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 3600)

# The two tutorials that had never been compared to OpenFOAM at all.
#
# motorBike is the big one -- 353700 cells from snappyHexMesh, kOmegaSST -- and the only tutorial here
# whose OpenFOAM side runs IN PARALLEL (decomposePar, snappyHexMesh and simpleFoam on 6 processors, then
# reconstructPar). Its ~5% on U is the LOOSEST agreement of any gated tutorial, and it is not a
# convergence artefact: run both codes 5x longer, to 2500, and the comparison is the same (U 5.58e-02
# against 5.00e-02, nut 9.56e-03 against 1.06e-02) while OpenFOAM's residual falls to 6.0e-05 and brae's
# to 1.4e-04. That is why the steady-state threshold is passed as 3e-04 here -- the extra iterations move
# the residual and not the answer -- and why the case is gated AS SHIPPED at its own endTime of 500.
# OpenFOAM's run is also potentialFoam-initialised and brae's is not, which is in the bounds too.
add_test(NAME tutorial_motorbike
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 motorBike U=8e-02,p=9e-02,k=1.7e-01,omega=1.0e-01,nut=2e-02 3e-04)
# known-red (2026-11-08, the user's decision): on the coarse mesh the cold start's pressure solver picks one of two steady states
# 9-10 % apart that both codes hold, OpenFOAM's GAMG one and brae's AMG-PCG the other (SIM-24, 2026-11-03), so the bounds are
# not met and stay as they are. The GB10 CI leaves the label out (.github/workflows/gpu-test.yml); ctest -L known-red runs it
set_tests_properties(tutorial_motorbike PROPERTIES SKIP_RETURN_CODE 77 LABELS "slow;known-red" TIMEOUT 5400)

# pitzDailyExptInlet ships no Allrun at all (blockMesh + simpleFoam) and its residualControl is very loose
# -- p 1e-2, U 1e-3. That looseness is most of the gap: at the case's own settings brae reads U 2.866e-02
# and p 1.630e-01 against OpenFOAM, and with both codes driven to ~1e-9 the same comparison reads
# U 1.155e-02 and p 1.540e-02. Gated as shipped, with the bounds that implies.
add_test(NAME tutorial_pitzdailyexptinlet
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 pitzDailyExptInlet U=4e-02,p=2.5e-01,k=1.2e-01,epsilon=1.0e-01,nut=1.8e-01)
set_tests_properties(tutorial_pitzdailyexptinlet PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

# squareBend, gated at 8000 rather than its shipped endTime of 500. That is not padding: at 500 the
# comparison reads U 2.759e-02 and at 8000 -- where both codes sit on the same 2.5e-05 residual plateau --
# it reads 1.773e-03. Gating the shipped number would record where the run stopped as if it were a
# property of the solver. 112000 cells, kEpsilon, SIMPLEC, `Gauss limitedLinearV 1`.
add_test(NAME tutorial_squarebend
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 squareBend U=5e-03,p=5e-03,k=1.2e-02,epsilon=3e-02,nut=5e-03 1e-04 8000)
set_tests_properties(tutorial_squarebend PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 3600)

# bump2D:kOmegaSST. U, p, k and omega agree to 5.3e-04, 5.2e-04, 9.4e-03 and 1.1e-05; nut alone sits at
# 1.1e-01, and that is the case's own residualControl of 1e-05 rather than the closure -- drive both codes
# to 1e-08 and the same comparison reads U 1.495e-05 and nut 9.905e-03, a factor of 11 on nut from nothing
# but the stopping point. Gated as shipped, since 40000 iterations is not a routine gate, with a nut bound
# that says so.
add_test(NAME tutorial_bump2d_komegasst
         COMMAND ${PROJECT_SOURCE_DIR}/tests/tutorial_vs_openfoam.sh
                 bump2D:kOmegaSST U=1.5e-03,p=1.5e-03,k=2.5e-02,omega=4e-05,nut=2.2e-01)
set_tests_properties(tutorial_bump2d_komegasst PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 2400)

# `limited <k> corrected` (OF limitedSnGrad) on both paths. Asserts `limited 1` == `corrected` bit-for-bit,
# that 0.33 actually bites, and that host and device agree -- the last of which caught the device limiting
# each velocity component independently where OpenFOAM shares one limiter across the vector.
add_test(NAME limitedsngrad_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/limitedsngrad_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(limitedsngrad_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# fvOptions explicitPorositySource/DarcyForchheimer end to end. The case is BUILT here (pitzDaily plus a
# topoSet cellZone) rather than using simpleCar, because brae disagrees with OpenFOAM on simpleCar by ~50%
# with fvOptions removed from both -- see the manifest. The control is a no-porosity run.
add_test(NAME fvoptions_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/fvoptions_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(fvoptions_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# fvOptions framework + porosity, unit level: the dictionary, the cellZone selection and the resistance
# expression, asserted against OpenFOAM's source and against what the case says.
add_test(NAME fvoptions_cpp
         COMMAND test_fvoptions_cpp ${PROJECT_SOURCE_DIR}/validation/simpleCarPorous 0)
set_tests_properties(fvoptions_cpp PROPERTIES SKIP_RETURN_CODE 77)

# limitedLinear / limitedLinearV / LUST end to end vs real OpenFOAM, each with a plain-upwind control
# run against the same oracle. The control asserts on k, not U: every one of these schemes lands within
# ~5.5e-03 of OpenFOAM on U and so does upwind, so U cannot discriminate.
add_test(NAME divschemes_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/divschemes_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(divschemes_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# linearUpwind vs real OpenFOAM on pitzDaily's own discretisation. Generates the oracle by running
# simpleFoam itself, and CONTROLS with a plain-upwind run: the deferred correction leaves the matrix
# untouched, so a port that dropped it converges and looks fine -- only the comparison can see it.
add_test(NAME linearupwind_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/linearupwind_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(linearupwind_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# kOmegaSST _cpp reference vs real OpenFOAM. The oracle is the RESIDUAL of our assembled equations at
# OpenFOAM's converged fields -- a statement about the discretisation -- with a perturbation control.
# The reference field set is generated by komegasst_vs_openfoam.sh; this test consumes it.
# limited/blended convection schemes: LUST, limitedLinear and its V variant. Asserted against exact
# properties from OpenFOAM's source (the TVD bound, both limiter limits, LUST's blend identity) with a
# control proving the V form couples the components rather than limiting them independently.
add_test(NAME limitedschemes_cpp
         COMMAND test_limitedschemes_cpp ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_test(NAME nonorth_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/nonorth_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/shearedChannel)
set_tests_properties(nonorth_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

add_test(NAME determinism_kepsilon
         COMMAND ${PROJECT_SOURCE_DIR}/tests/determinism_gate.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily 20)
set_tests_properties(determinism_kepsilon PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)
add_test(NAME determinism_komega
         COMMAND ${PROJECT_SOURCE_DIR}/tests/determinism_gate.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailyKOmega 20)
set_tests_properties(determinism_komega PROPERTIES SKIP_RETURN_CODE 77 LABELS slow)

# TURBULENT gate for the rebuilt loop: one k-epsilon SIMPLE iteration from OpenFOAM's CONVERGED solution
# must leave it there, with a laminar control proving nut really reaches the momentum equation.
add_executable(test_simple_turbulent_cpp ${PROJECT_SOURCE_DIR}/tests/test_simple_turbulent_cpp.cu)
target_include_directories(test_simple_turbulent_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_turbulent_cpp PRIVATE brae_core)

add_executable(test_celllimited_cpp ${PROJECT_SOURCE_DIR}/tests/test_celllimited_cpp.cu)
target_include_directories(test_celllimited_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_celllimited_cpp PRIVATE brae_core)

add_executable(actuationdisk_probe ${PROJECT_SOURCE_DIR}/tests/actuationdisk_probe.cu)
target_include_directories(actuationdisk_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(actuationdisk_probe PRIVATE brae_core)
add_test(NAME celllimited_cpp
         COMMAND test_celllimited_cpp ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(mrf_probe ${PROJECT_SOURCE_DIR}/tests/mrf_probe.cu)
target_include_directories(mrf_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(mrf_probe PRIVATE brae_core)
add_test(NAME simple_turbulent_cpp
         COMMAND test_simple_turbulent_cpp ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_simple_turbulent ${PROJECT_SOURCE_DIR}/tests/test_simple_turbulent.cu)
target_include_directories(test_simple_turbulent PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_simple_turbulent PRIVATE brae_core)
# Fast fidelity gates (sub-second): one cf turbulent step from OpenFOAM's converged solution must
# reproduce it to OF's convergence floor. "cap-and-measure", no thousand-iteration runs needed.
add_test(NAME turb_fixedpoint_12k
         COMMAND test_simple_turbulent ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb     1576 1576 1)
add_test(NAME turb_fixedpoint_49k
         COMMAND test_simple_turbulent ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurbBig   3000 3000 1)
# Full end-to-end convergence from 0/ (~2.5 min/12k); slow-labelled so the fast loop skips it
# (`ctest -LE slow`); run explicitly with `ctest -R simple_turbulent_full`.
add_test(NAME simple_turbulent_full
         COMMAND test_simple_turbulent ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 0 1576)
set_tests_properties(simple_turbulent_full PROPERTIES LABELS slow TIMEOUT 900)

add_test(NAME simple_piov_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/simple_piov_vs_openfoam.sh)
set_tests_properties(simple_piov_vs_openfoam PROPERTIES
                     ENVIRONMENT "BUILD=${PROJECT_BINARY_DIR}"
                     SKIP_RETURN_CODE 77
                     DEPENDS brae
                     TIMEOUT 1800)

add_executable(test_residual_control ${PROJECT_SOURCE_DIR}/tests/test_residual_control.cu)
target_include_directories(test_residual_control PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_residual_control PRIVATE brae_core)
add_test(NAME residual_control COMMAND test_residual_control)

add_executable(test_fvoptions_sources_key ${PROJECT_SOURCE_DIR}/tests/test_fvoptions_sources_key.cu)
target_include_directories(test_fvoptions_sources_key PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_fvoptions_sources_key PRIVATE brae_core)
add_test(NAME fvoptions_sources_key COMMAND test_fvoptions_sources_key)

# MRF increment M1: MRFZone frame flux + Coriolis source (serial)
add_executable(test_mrf_zone ${PROJECT_SOURCE_DIR}/tests/test_mrf_zone.cu)
target_include_directories(test_mrf_zone PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mrf_zone PRIVATE brae_core)
add_test(NAME mrf_zone COMMAND test_mrf_zone ${PROJECT_SOURCE_DIR}/validation/mrfBox)

# MRF increment M2+M3: MRF in the full SIMPLE loop (solid-body rotation)
add_executable(test_mrf_rotation ${PROJECT_SOURCE_DIR}/tests/test_mrf_rotation.cu)
target_include_directories(test_mrf_rotation PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_mrf_rotation PRIVATE brae_core)
add_test(NAME mrf_rotation COMMAND test_mrf_rotation ${PROJECT_SOURCE_DIR}/validation/mrfBox)

# GPU offload G5: device pressure equation (assemble + fold + solve end-to-end)
add_executable(test_gpu_pressure ${PROJECT_SOURCE_DIR}/tests/test_gpu_pressure.cu)
target_include_directories(test_gpu_pressure PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_pressure PRIVATE brae_core)
add_test(NAME gpu_pressure COMMAND test_gpu_pressure ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G6: device momentum predictor (assemble div-laplacian + 3-component BiCGStab solve)
add_executable(test_gpu_momentum ${PROJECT_SOURCE_DIR}/tests/test_gpu_momentum.cu)
target_include_directories(test_gpu_momentum PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_momentum PRIVATE brae_core)
add_test(NAME gpu_momentum COMMAND test_gpu_momentum ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G7: SIMPLE coupling glue on device (H/rAU/phiHbyA/matrixFlux/corrector)
# CUDA vs the _cpp REFERENCE at stage granularity. Closes the chain OpenFOAM -> _cpp -> CUDA: every other
# GPU test compares the device against CPU code written inline in that same test, which proves consistency
# but not correctness.
add_executable(test_gpu_vs_cpp ${PROJECT_SOURCE_DIR}/tests/test_gpu_vs_cpp.cu)
target_include_directories(test_gpu_vs_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_vs_cpp PRIVATE brae_core)
add_test(NAME gpu_vs_cpp
         COMMAND test_gpu_vs_cpp ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym 282)
# ...and again on a TURBULENT case, where nuEff varies per cell and per boundary face. On the laminar case
# above nuEff is constant, so a kernel that mishandles a per-face diffusivity -- or reads the owner cell's
# value on a wall instead of the patch value -- would still agree perfectly.
add_test(NAME gpu_vs_cpp_turbulent
         COMMAND test_gpu_vs_cpp ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb 1576)

add_executable(test_gpu_simple ${PROJECT_SOURCE_DIR}/tests/test_gpu_simple.cu)
target_include_directories(test_gpu_simple PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_simple PRIVATE brae_core)
add_test(NAME gpu_simple COMMAND test_gpu_simple ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: the closed device SIMPLE step (orchestration of G3-G7)
add_executable(test_gpu_step ${PROJECT_SOURCE_DIR}/tests/test_gpu_step.cu)
target_include_directories(test_gpu_step PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_step PRIVATE brae_core)
add_test(NAME gpu_step COMMAND test_gpu_step ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: on-device boundary-condition evaluation
add_executable(test_gpu_boundary ${PROJECT_SOURCE_DIR}/tests/test_gpu_boundary.cu)
target_include_directories(test_gpu_boundary PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_boundary PRIVATE brae_core)
add_test(NAME gpu_boundary COMMAND test_gpu_boundary ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: on-device vector BC evaluation (momentum)
add_executable(test_gpu_boundary_vec ${PROJECT_SOURCE_DIR}/tests/test_gpu_boundary_vec.cu)
target_include_directories(test_gpu_boundary_vec PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_boundary_vec PRIVATE brae_core)
add_test(NAME gpu_boundary_vec COMMAND test_gpu_boundary_vec ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: boundary-flux kernels (phiHbyA.boundary, pEqn.flux().boundary)
add_executable(test_gpu_bndflux ${PROJECT_SOURCE_DIR}/tests/test_gpu_bndflux.cu)
target_include_directories(test_gpu_bndflux PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_bndflux PRIVATE brae_core)
add_test(NAME gpu_bndflux COMMAND test_gpu_bndflux ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: the fully device-resident SIMPLE loop (residency capstone)
add_executable(test_gpu_resident ${PROJECT_SOURCE_DIR}/tests/test_gpu_resident.cu)
target_include_directories(test_gpu_resident PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_resident PRIVATE brae_core)
add_test(NAME gpu_resident COMMAND test_gpu_resident ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: device matrix relaxation (relaxMatrix)
add_executable(test_gpu_relax ${PROJECT_SOURCE_DIR}/tests/test_gpu_relax.cu)
target_include_directories(test_gpu_relax PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_relax PRIVATE brae_core)
add_test(NAME gpu_relax COMMAND test_gpu_relax ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload: the TURBULENT device-resident driver (SIMPLE + k-epsilon, fully on GPU)
add_executable(test_gpu_resident_turb ${PROJECT_SOURCE_DIR}/tests/test_gpu_resident_turb.cu)
target_include_directories(test_gpu_resident_turb PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_resident_turb PRIVATE brae_core)
add_test(NAME gpu_resident_turb COMMAND test_gpu_resident_turb ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

add_executable(test_gpu_benchmark ${PROJECT_SOURCE_DIR}/tests/test_gpu_benchmark.cu)
target_include_directories(test_gpu_benchmark PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_benchmark PRIVATE brae_core)
add_test(NAME gpu_benchmark COMMAND test_gpu_benchmark ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload #4: the device-resident driver (DeviceSimpleSolver) vs the CPU runSimple oracle, real case.
add_executable(test_gpu_driver ${PROJECT_SOURCE_DIR}/tests/test_gpu_driver.cu)
target_include_directories(test_gpu_driver PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_driver PRIVATE brae_core)
add_test(NAME gpu_driver COMMAND test_gpu_driver ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
