# Dictionaries, fields and Function1. Reading and writing what OpenFOAM writes: dictionary expansion (#include, #eval,
# scoped macros), binary and gzip fields, Function1 and coded/expression entries, function-object and fvOptions
# dictionaries.
#
# Included by tests/CMakeLists.txt.

add_executable(test_gpu_coded_bc ${PROJECT_SOURCE_DIR}/tests/test_gpu_coded_bc.cu)
target_include_directories(test_gpu_coded_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_coded_bc PRIVATE brae_core)
add_test(NAME gpu_coded_bc COMMAND test_gpu_coded_bc)

# OpenFOAM's `type coded;` Function1, host side: the snippet compiles and loads, x reaches it, a
# function-local static lives for the process, and the keys brae cannot reproduce are refused by name.
# No OpenFOAM oracle here -- that is rho_coded_function1_vs_openfoam, which this covers underneath and
# which exits 77 where OpenFOAM is not installed.
add_executable(test_coded_function1 ${PROJECT_SOURCE_DIR}/tests/test_coded_function1.cu)
target_include_directories(test_coded_function1 PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_coded_function1 PRIVATE brae_core)
add_test(NAME coded_function1 COMMAND test_coded_function1)

# OpenFOAM's `type expression;` PatchFunction1 evaluator, host side, without OpenFOAM: the grammar's
# precedence and associativity, the field lookups in OpenFOAM's order, Foam::max's NaN side, mag() of a
# vector, and every token outside the carried subset refused by name. The OpenFOAM oracle is
# rho_patch_expression_vs_openfoam, which exits 77 where OpenFOAM is not installed.
add_executable(test_patch_expr ${PROJECT_SOURCE_DIR}/tests/test_patch_expr.cu)
target_include_directories(test_patch_expr PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_patch_expr PRIVATE brae_core)
add_test(NAME patch_expr COMMAND test_patch_expr)

add_executable(test_fields_bc ${PROJECT_SOURCE_DIR}/tests/test_fields_bc.cu)
target_include_directories(test_fields_bc PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_fields_bc PRIVATE brae_core)
add_test(NAME fields_bc
         COMMAND test_fields_bc ${PROJECT_SOURCE_DIR}/validation/pitzDaily282)

add_executable(test_tvm_mapping ${PROJECT_SOURCE_DIR}/tests/test_tvm_mapping.cu)
target_include_directories(test_tvm_mapping PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_tvm_mapping PRIVATE brae_core)
add_test(NAME tvm_mapping COMMAND test_tvm_mapping)

add_executable(test_foam_constants ${PROJECT_SOURCE_DIR}/tests/test_foam_constants.cu)
target_include_directories(test_foam_constants PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_foam_constants PRIVATE brae_core)
add_test(NAME foam_constants COMMAND test_foam_constants)

# Group C: fvSchemes entries must resolve within their own sub-dictionary (C1-C4).
add_executable(test_scheme_blocks ${PROJECT_SOURCE_DIR}/tests/test_scheme_blocks.cu)
target_include_directories(test_scheme_blocks PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_scheme_blocks PRIVATE brae_core)
add_test(NAME scheme_blocks COMMAND test_scheme_blocks)

add_executable(test_overset_refused ${PROJECT_SOURCE_DIR}/tests/test_overset_refused.cu)
target_include_directories(test_overset_refused PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_overset_refused PRIVATE brae_core)
add_executable(test_uniform_function1 ${PROJECT_SOURCE_DIR}/tests/test_uniform_function1.cu)
target_include_directories(test_uniform_function1 PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_uniform_function1 PRIVATE brae_core)
add_test(NAME uniform_function1 COMMAND test_uniform_function1)

add_test(NAME overset_refused
         COMMAND test_overset_refused ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

add_executable(test_boundary_subdict_skip ${PROJECT_SOURCE_DIR}/tests/test_boundary_subdict_skip.cu)
target_include_directories(test_boundary_subdict_skip PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_boundary_subdict_skip PRIVATE brae_core)
add_test(NAME boundary_subdict_skip COMMAND test_boundary_subdict_skip)

add_executable(test_dict_scoped_macro ${PROJECT_SOURCE_DIR}/tests/test_dict_scoped_macro.cu)
target_include_directories(test_dict_scoped_macro PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_dict_scoped_macro PRIVATE brae_core)
add_test(NAME dict_scoped_macro COMMAND test_dict_scoped_macro)

add_executable(test_eval_scoped ${PROJECT_SOURCE_DIR}/tests/test_eval_scoped.cu)
target_include_directories(test_eval_scoped PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_eval_scoped PRIVATE brae_core)
add_test(NAME eval_scoped COMMAND test_eval_scoped)

add_executable(test_hbya_inletoutlet ${PROJECT_SOURCE_DIR}/tests/test_hbya_inletoutlet.cu)
target_include_directories(test_hbya_inletoutlet PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_hbya_inletoutlet PRIVATE brae_core)
add_test(NAME hbya_inletoutlet COMMAND test_hbya_inletoutlet)

add_executable(test_gz_field_write ${PROJECT_SOURCE_DIR}/tests/test_gz_field_write.cu)
target_include_directories(test_gz_field_write PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_gz_field_write PRIVATE brae_core)
add_test(NAME gz_field_write COMMAND test_gz_field_write)

add_executable(test_fan_pressure ${PROJECT_SOURCE_DIR}/tests/test_fan_pressure.cu)
target_include_directories(test_fan_pressure PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_fan_pressure PRIVATE brae_core)
add_test(NAME fan_pressure COMMAND test_fan_pressure)

add_executable(test_binary_field_read ${PROJECT_SOURCE_DIR}/tests/test_binary_field_read.cu)
target_include_directories(test_binary_field_read PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_binary_field_read PRIVATE brae_core)
add_test(NAME binary_field_read COMMAND test_binary_field_read)
# nutUSpaldingWallFunction's own tolerance and maxIter, and calcNut's keep branch (brief 2026-11-11 #18)
add_executable(test_nut_spalding_controls ${PROJECT_SOURCE_DIR}/tests/test_nut_spalding_controls.cu)
target_include_directories(test_nut_spalding_controls PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_nut_spalding_controls PRIVATE brae_core)
add_test(NAME nut_spalding_controls COMMAND test_nut_spalding_controls)

add_executable(test_fixed_mean ${PROJECT_SOURCE_DIR}/tests/test_fixed_mean.cu)
target_include_directories(test_fixed_mean PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_fixed_mean PRIVATE brae_core)
add_test(NAME fixed_mean COMMAND test_fixed_mean)
