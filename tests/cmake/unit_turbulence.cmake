# Turbulence models. kEpsilon, realizableKE, RNG, kOmegaSST(+LM, DDES, IDDES), Spalart-Allmaras (DDES, IDDES), LES
# models and deltas, wall functions and nut, on the host reference and the device closures.
#
# Included by tests/CMakeLists.txt.

add_executable(test_cell_wall_dist ${PROJECT_SOURCE_DIR}/tests/test_cell_wall_dist.cu)
target_include_directories(test_cell_wall_dist PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_cell_wall_dist PRIVATE brae_core)
add_test(NAME cell_wall_dist
         COMMAND test_cell_wall_dist ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

add_executable(test_komega_coeffs ${PROJECT_SOURCE_DIR}/tests/test_komega_coeffs.cu)
target_include_directories(test_komega_coeffs PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_komega_coeffs PRIVATE brae_core)
add_test(NAME komega_coeffs
         COMMAND test_komega_coeffs ${PROJECT_SOURCE_DIR}/validation/komegaCoeffs)

add_executable(test_gpu_komega_sst ${PROJECT_SOURCE_DIR}/tests/test_gpu_komega_sst.cu)
target_include_directories(test_gpu_komega_sst PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_komega_sst PRIVATE brae_core)
add_test(NAME gpu_komega_sst
         COMMAND test_gpu_komega_sst ${PROJECT_SOURCE_DIR}/validation/pitzDaily282/komega_sst.dat)

# SA-DDES length scale (DES/LES): deviceSADDESdTilda reproduces the OF SpalartAllmarasDDES delayed length scale.
add_executable(test_gpu_saddes ${PROJECT_SOURCE_DIR}/tests/test_gpu_saddes.cu)
target_include_directories(test_gpu_saddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_saddes PRIVATE brae_core)
add_test(NAME gpu_saddes COMMAND test_gpu_saddes)

# kOmegaSST-DDES DES factor: deviceKOmegaSSTDESfactor reproduces the OF kOmegaSSTDDES k-destruction enhancement.
add_executable(test_gpu_ksstddes ${PROJECT_SOURCE_DIR}/tests/test_gpu_ksstddes.cu)
target_include_directories(test_gpu_ksstddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ksstddes PRIVATE brae_core)
add_test(NAME gpu_ksstddes COMMAND test_gpu_ksstddes)

add_executable(test_gpu_smagorinsky ${PROJECT_SOURCE_DIR}/tests/test_gpu_smagorinsky.cu)
target_include_directories(test_gpu_smagorinsky PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_smagorinsky PRIVATE brae_core)
add_test(NAME gpu_smagorinsky COMMAND test_gpu_smagorinsky)

# SA-IDDES against REAL OpenFOAM's SpalartAllmarasIDDES: the length scale's formula, the defaults and the reader,
# on a dump of OpenFOAM's own dTilda() (tools/dumpSAIDDES, built with wmake; the gates skip without it).
add_executable(test_sa_iddes_vs_openfoam ${PROJECT_SOURCE_DIR}/tests/test_sa_iddes_vs_openfoam.cu)
target_include_directories(test_sa_iddes_vs_openfoam PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_sa_iddes_vs_openfoam PRIVATE brae_core)
foreach(SA_IDDES_MODE kernel defaults refusals)
    add_test(NAME sa_iddes_${SA_IDDES_MODE}_vs_openfoam
             COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_iddes_${SA_IDDES_MODE}_vs_openfoam.sh)
    set_tests_properties(sa_iddes_${SA_IDDES_MODE}_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
add_executable(test_gpu_saiddes ${PROJECT_SOURCE_DIR}/tests/test_gpu_saiddes.cu)
target_include_directories(test_gpu_saiddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_saiddes PRIVATE brae_core)
add_test(NAME gpu_saiddes COMMAND test_gpu_saiddes)

add_executable(test_gpu_ksstiddes ${PROJECT_SOURCE_DIR}/tests/test_gpu_ksstiddes.cu)
target_include_directories(test_gpu_ksstiddes PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_ksstiddes PRIVATE brae_core)
add_test(NAME gpu_ksstiddes COMMAND test_gpu_ksstiddes)

add_executable(test_wall_function ${PROJECT_SOURCE_DIR}/tests/test_wall_function.cu)
target_include_directories(test_wall_function PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_wall_function PRIVATE brae_core)
add_test(NAME wall_function
         COMMAND test_wall_function ${PROJECT_SOURCE_DIR}/validation/pitzDaily282)

# Audit finding #14: the algebraic-LES path honoured only nutUSpalding while printing any other nut
# wall function as honoured, labelled "kEpsilon" whatever the model. Non-Spalding wall functions on
# LES now refuse by name, and every wall-fn print carries the model word the case names.
add_test(NAME les_wallfunction_refusals COMMAND ${PROJECT_SOURCE_DIR}/tests/les_wallfunction_refusals.sh)
set_tests_properties(les_wallfunction_refusals PROPERTIES SKIP_RETURN_CODE 77)

# Audit finding #15: the SA device path hard-forced Spalding whatever 0/nut asked for --
# bump2D:SpalartAllmaras ships nutLowReWallFunction (calcNut() = Zero on every model) and got a Newton
# uTau inside a green gate. The BC selects now: Spalding/LowRe honoured, k-based family and plain wall
# BCs refused by name, two different wall functions refused.
add_test(NAME sa_wallfunction_refusals COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_wallfunction_refusals.sh)
set_tests_properties(sa_wallfunction_refusals PROPERTIES SKIP_RETURN_CODE 77)

# The RAS model by its keyword on the simpleFoam drivers: `model`, and `RASModel` only where that is
# absent (RASModel.C:140-144). `RAS { model SpalartAllmaras; }` was refused as an empty RASModel. Byte
# identity against the `RASModel` case on both drivers, OpenFOAM's own premise, and a compat-only control.
add_test(NAME sa_turbulence_model_keyword COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_turbulence_model_keyword.sh)
set_tests_properties(sa_turbulence_model_keyword PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_komegasst_cpp ${PROJECT_SOURCE_DIR}/tests/test_komegasst_cpp.cu)
target_include_directories(test_komegasst_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_komegasst_cpp PRIVATE brae_core)

add_executable(test_realizablke_cpp ${PROJECT_SOURCE_DIR}/tests/test_realizablke_cpp.cu)
target_include_directories(test_realizablke_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_realizablke_cpp PRIVATE brae_core)

# realizableKE's rCmu on the device against the host reference at strain rates where the constant in its
# denominator (OpenFOAM's SMALL) decides the answer; the control is the kernel's own switch, run as a child.
add_executable(test_device_realizable_strain ${PROJECT_SOURCE_DIR}/tests/test_device_realizable_strain.cu)
target_include_directories(test_device_realizable_strain PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_device_realizable_strain PRIVATE brae_core)
add_test(NAME device_realizable_strain COMMAND test_device_realizable_strain)
set_tests_properties(device_realizable_strain PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_kepsilon ${PROJECT_SOURCE_DIR}/tests/test_kepsilon.cu)
target_include_directories(test_kepsilon PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_kepsilon PRIVATE brae_core)
add_test(NAME kepsilon
         COMMAND test_kepsilon ${PROJECT_SOURCE_DIR}/validation/pitzDaily282)

add_executable(test_kepsilon_eqn ${PROJECT_SOURCE_DIR}/tests/test_kepsilon_eqn.cu)
target_include_directories(test_kepsilon_eqn PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_kepsilon_eqn PRIVATE brae_core)
add_test(NAME kepsilon_eqn
         COMMAND test_kepsilon_eqn ${PROJECT_SOURCE_DIR}/validation/kEpsCase)

add_executable(test_kepsilon_correct ${PROJECT_SOURCE_DIR}/tests/test_kepsilon_correct.cu)
target_include_directories(test_kepsilon_correct PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_kepsilon_correct PRIVATE brae_core)
add_test(NAME kepsilon_correct
         COMMAND test_kepsilon_correct ${PROJECT_SOURCE_DIR}/validation/kEpsCorrect)

# The k/epsilon PATCH diffusivity: OF's `calculated` nut patches carry Cmu*k_b^2/eps_b, not the cell value,
# and DkEff(patchi)/DepsilonEff(patchi) are built from it. Started from OpenFOAM's own converged fields so
# the initial residual measures the discretisation directly; needs no OpenFOAM install, only its log.
add_test(NAME boundary_nut_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/boundary_nut_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurb)
set_tests_properties(boundary_nut_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

# kOmegaSST on the _cpp path vs real OpenFOAM. The fast entry probes one iteration from OpenFOAM's own
# converged state and carries its own upwind negative control; the slow one runs the case end to end from
# 0/, which is what exposed the omega blow-up that the probe could not see.
add_test(NAME sst_cpp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sst_cpp_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailySST)
set_tests_properties(sst_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME sst_cpp_full
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sst_cpp_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailySST full)
set_tests_properties(sst_cpp_full PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1200)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one module at a time. The fast entry
# also asserts brae RESOLVES div(phi,k) to `bounded limitedLinear` rather than silently running upwind, and
# is what catches the wall mask: keying it on epsilonWallFunction left every kOmegaSST case with no wall
# faces at all, measuring U 48x and omega 6.8x off OpenFOAM.
add_test(NAME sst_cuda_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sst_cuda_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailySST)
set_tests_properties(sst_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME sst_cuda_full
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sst_cuda_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/pitzDailySST full)
set_tests_properties(sst_cuda_full PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# SpalartAllmaras end to end on the _cpp path vs real OpenFOAM, with a laminar control. This is the case
# that exposed nut's wall value being taken from the nut_ = nuTilda*fv1 assignment instead of
# nutUSpaldingWallFunction -- see the script header for the three defects it cost.
add_test(NAME sa_cpp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_cpp_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(sa_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 900)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one module at a time. Its cold start
# is what the fixed-point comparison could not see: the two momentum assemblies are bit-identical, and the
# whole gap was the Spalding wall Newton being cold-started instead of seeded from nut_ -- see the header.
add_test(NAME sa_cuda_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_cuda_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/airFoil2D)
set_tests_properties(sa_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# kOmegaSSTLM (Langtry-Menter transition) on the _cpp REFERENCE, run END TO END on the real T3A tutorial
# against real OpenFOAM. Its control is the same case with the transition model removed -- 167x on U and
# 437x on k -- which is what makes the bounds above mean something.
add_test(NAME lm_cpp_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/lm_cpp_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A 269 500)
set_tests_properties(lm_cpp_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

# The SAME case through the CUDA V2 driver, ported onto the _cpp path one module at a time. The module
# comparison (lm_cuda_probe: Fthetat, both reaction preps and gammaIntEff to 1e-12) runs FIRST inside the
# gate, so an end-to-end agreement that came from cancellation cannot pass it. Control: plain kOmegaSST.
add_test(NAME lm_cuda_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/lm_cuda_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A 269)
set_tests_properties(lm_cuda_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1800)

add_test(NAME closure_gradlimit_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/closure_gradlimit_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/windAroundBuildingsBox)
set_tests_properties(closure_gradlimit_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 LABELS slow TIMEOUT 1200)

add_test(NAME sa_coeffs_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/sa_coeffs_vs_openfoam.sh)
set_tests_properties(sa_coeffs_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# realizableKE _cpp reference vs real OpenFOAM. Same residual oracle as kOmegaSST, plus the two controls
# that say WHICH model was ported: rCmu must genuinely vary (a constant 0.09 is standard k-epsilon, which
# would still work), and S2 must use devSymm rather than the symm that kOmegaSST uses.
add_test(NAME realizablke_cpp
         COMMAND test_realizablke_cpp ${PROJECT_SOURCE_DIR}/validation/pitzDailyRKE 2000)
set_tests_properties(realizablke_cpp PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME komegasst_cpp
         COMMAND test_komegasst_cpp ${PROJECT_SOURCE_DIR}/validation/pitzDailySST 2000)
set_tests_properties(komegasst_cpp PROPERTIES SKIP_RETURN_CODE 77)

# The _cpp reference for divDevReff, proven against the SAME OpenFOAM dump as the inline test above,
# with a wrong-sign check and a wall-nuEff negative control so the test is known to be able to fail.
add_executable(test_divdevreff_cpp ${PROJECT_SOURCE_DIR}/tests/test_divdevreff_cpp.cu)
target_include_directories(test_divdevreff_cpp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_divdevreff_cpp PRIVATE brae_core)
add_test(NAME divdevreff_cpp
         COMMAND test_divdevreff_cpp ${PROJECT_SOURCE_DIR}/validation/kEpsCorrect)

add_executable(test_divdevreff ${PROJECT_SOURCE_DIR}/tests/test_divdevreff.cu)
target_include_directories(test_divdevreff PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_divdevreff PRIVATE brae_core)
add_test(NAME divdevreff
         COMMAND test_divdevreff ${PROJECT_SOURCE_DIR}/validation/kEpsCorrect)

add_executable(lm_cuda_probe ${PROJECT_SOURCE_DIR}/tests/lm_cuda_probe.cu)
target_include_directories(lm_cuda_probe PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(lm_cuda_probe PRIVATE brae_core)

# End-to-end kOmegaSSTLM (Langtry-Menter transition) vs a SAVED OpenFOAM v2412 oracle (no OF needed at test time):
# runs brae on a transition flat plate and checks U L2 < 3% and the gammaInt transition is captured.
# slow-labelled (~2 min); regenerate the oracle with validation/legacy_cf/komegasstlm_wall_vs_openfoam.sh -- QUARANTINED (D5): it's still on retired cf paths and cannot run as-is.
add_test(NAME komegasstlm_flatplate
         COMMAND bash ${PROJECT_SOURCE_DIR}/validation/lm_ctest.sh
                 $<TARGET_FILE:brae> ${PROJECT_SOURCE_DIR}/validation/lmFlatPlate)
set_tests_properties(komegasstlm_flatplate PROPERTIES LABELS slow TIMEOUT 600)

add_executable(test_wall_data_refresh ${PROJECT_SOURCE_DIR}/tests/test_wall_data_refresh.cu)
target_include_directories(test_wall_data_refresh PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_wall_data_refresh PRIVATE brae_core)
add_test(NAME wall_data_refresh COMMAND test_wall_data_refresh)
set_tests_properties(wall_data_refresh PROPERTIES ENVIRONMENT "BRAE_ALLOW_ACMI=1")

add_executable(test_kepsilon_production_and_blend ${PROJECT_SOURCE_DIR}/tests/test_kepsilon_production_and_blend.cu)
target_include_directories(test_kepsilon_production_and_blend PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_kepsilon_production_and_blend PRIVATE brae_core)
add_test(NAME kepsilon_production_and_blend COMMAND test_kepsilon_production_and_blend)

add_executable(test_near_wall_dist_combined ${PROJECT_SOURCE_DIR}/tests/test_near_wall_dist_combined.cu)
target_include_directories(test_near_wall_dist_combined PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_near_wall_dist_combined PRIVATE brae_core)
add_test(NAME near_wall_dist_combined COMMAND test_near_wall_dist_combined)

add_executable(test_rng_kepsilon ${PROJECT_SOURCE_DIR}/tests/test_rng_kepsilon.cu)
target_include_directories(test_rng_kepsilon PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_rng_kepsilon PRIVATE brae_core)
add_test(NAME rng_kepsilon COMMAND test_rng_kepsilon)

add_executable(test_maxwell_terms ${PROJECT_SOURCE_DIR}/tests/test_maxwell_terms.cu)
target_include_directories(test_maxwell_terms PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_maxwell_terms PRIVATE brae_core)
add_test(NAME maxwell_terms COMMAND test_maxwell_terms)

add_executable(test_zdes_shielding ${PROJECT_SOURCE_DIR}/tests/test_zdes_shielding.cu)
target_include_directories(test_zdes_shielding PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_zdes_shielding PRIVATE brae_core)
add_test(NAME zdes_shielding COMMAND test_zdes_shielding)

add_executable(test_max_delta_xyz ${PROJECT_SOURCE_DIR}/tests/test_max_delta_xyz.cu)
target_include_directories(test_max_delta_xyz PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_max_delta_xyz PRIVATE brae_core)
add_test(NAME max_delta_xyz COMMAND test_max_delta_xyz)

add_executable(test_wale_nut ${PROJECT_SOURCE_DIR}/tests/test_wale_nut.cu)
target_include_directories(test_wale_nut PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_wale_nut PRIVATE brae_core)
add_test(NAME wale_nut COMMAND test_wale_nut)

add_executable(test_exact_wall_dist ${PROJECT_SOURCE_DIR}/tests/test_exact_wall_dist.cu)
target_include_directories(test_exact_wall_dist PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_exact_wall_dist PRIVATE brae_core)
add_test(NAME exact_wall_dist COMMAND test_exact_wall_dist)

add_executable(test_deshybrid ${PROJECT_SOURCE_DIR}/tests/test_deshybrid.cu)
target_include_directories(test_deshybrid PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_deshybrid PRIVATE brae_core)
add_test(NAME deshybrid COMMAND test_deshybrid)

# divDevReff must use the case's NAMED grad(U) scheme, not a bare Gauss one (see the test's header).
add_executable(test_divdevreff_gradscheme ${PROJECT_SOURCE_DIR}/tests/test_divdevreff_gradscheme.cu)
target_include_directories(test_divdevreff_gradscheme PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_divdevreff_gradscheme PRIVATE brae_core)
add_test(NAME divdevreff_gradscheme COMMAND test_divdevreff_gradscheme)

# GPU offload (#2): explicit divDevReff stress source on device
add_executable(test_gpu_divdevreff ${PROJECT_SOURCE_DIR}/tests/test_gpu_divdevreff.cu)
target_include_directories(test_gpu_divdevreff PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_divdevreff PRIVATE brae_core)
add_test(NAME gpu_divdevreff COMMAND test_gpu_divdevreff ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload (#3): k-epsilon production + eddy viscosity on device
add_executable(test_gpu_kepsilon ${PROJECT_SOURCE_DIR}/tests/test_gpu_kepsilon.cu)
target_include_directories(test_gpu_kepsilon PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_kepsilon PRIVATE brae_core)
add_test(NAME gpu_kepsilon COMMAND test_gpu_kepsilon ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload (#3): closed device kEpsilon::correct() with setValues
add_executable(test_gpu_kepsilon_correct ${PROJECT_SOURCE_DIR}/tests/test_gpu_kepsilon_correct.cu)
target_include_directories(test_gpu_kepsilon_correct PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_kepsilon_correct PRIVATE brae_core)
add_test(NAME gpu_kepsilon_correct COMMAND test_gpu_kepsilon_correct ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
