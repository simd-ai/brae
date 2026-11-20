# Multi-rank (MPI) tests. Interface exchange, rank-GPU binding and decomposition. Names ending _np<N> take PROCESSORS
# N and, for N > 1, RESOURCE_LOCK gpu_multirank (tests/CMakeLists.txt).
#
# Included by tests/CMakeLists.txt.

add_executable(test_interface_exchange ${PROJECT_SOURCE_DIR}/tests/test_interface_exchange.cu)
target_include_directories(test_interface_exchange PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_interface_exchange PRIVATE brae_core)

foreach(_np 1 2 4 8)
    add_test(NAME interface_exchange_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_interface_exchange>)
endforeach()

# Multi-GPU Phase 0a: rank <-> GPU binding in Pstream::init
add_executable(test_gpu_bind ${PROJECT_SOURCE_DIR}/tests/test_gpu_bind.cu)
target_include_directories(test_gpu_bind PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_bind PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME gpu_bind_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_gpu_bind>)
endforeach()

# Multi-GPU Phase 0b: per-rank DeviceMesh upload + device-path reconstruction
add_executable(test_gpu_decompose ${PROJECT_SOURCE_DIR}/tests/test_gpu_decompose.cu)
target_include_directories(test_gpu_decompose PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_decompose PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME gpu_decompose_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_gpu_decompose>
                     ${PROJECT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

# Multi-GPU Phase 1a: NVSHMEM bootstrap + symmetric heap (host-side) smoke test
# ---------------------------------------------------------------------------------------------------
# The multi-GPU (distributed) test suite lived here, guarded by if(BRAE_WITH_NVSHMEM). It moved to
# legacy/tests/ along with the distributed solvers -- multi-GPU is OUT OF SCOPE. See legacy/README.md.
# Nothing in legacy/ is built or registered; do not re-add it without an explicit decision to resume.
# ---------------------------------------------------------------------------------------------------

add_executable(test_decompose_exchange ${PROJECT_SOURCE_DIR}/tests/test_decompose_exchange.cu)
target_include_directories(test_decompose_exchange PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_decompose_exchange PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME decompose_exchange_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_decompose_exchange>
                     ${PROJECT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_local_mesh ${PROJECT_SOURCE_DIR}/tests/test_local_mesh.cu)
target_include_directories(test_local_mesh PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_local_mesh PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME local_mesh_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_local_mesh>
                     ${PROJECT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_proc_delta ${PROJECT_SOURCE_DIR}/tests/test_proc_delta.cu)
target_include_directories(test_proc_delta PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_proc_delta PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME proc_delta_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_proc_delta>
                     ${PROJECT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_local_assembly ${PROJECT_SOURCE_DIR}/tests/test_local_assembly.cu)
target_include_directories(test_local_assembly PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_local_assembly PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME local_assembly_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_local_assembly>
                     ${PROJECT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_local_convection ${PROJECT_SOURCE_DIR}/tests/test_local_convection.cu)
target_include_directories(test_local_convection PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_local_convection PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME local_convection_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_local_convection>
                     ${PROJECT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

add_executable(test_processor_field ${PROJECT_SOURCE_DIR}/tests/test_processor_field.cu)
target_include_directories(test_processor_field PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_processor_field PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME processor_field_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_processor_field>
                     ${PROJECT_SOURCE_DIR}/validation/pitzDaily/constant/polyMesh)
endforeach()

add_executable(test_proc_interp ${PROJECT_SOURCE_DIR}/tests/test_proc_interp.cu)
target_include_directories(test_proc_interp PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_proc_interp PRIVATE brae_core)
foreach(_np 1 2 4 8)
    add_test(NAME proc_interp_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np ${_np}
                     $<TARGET_FILE:test_proc_interp>
                     ${PROJECT_SOURCE_DIR}/validation/kEpsCorrect)
endforeach()

# Converged-run driver (not a ctest gate): mpirun -np N test_parallel_converged <case> <ofTime> <iters>

# brae_simpleFoam (the MPI-parallel HOST solver) moved to legacy/ -- multi-GPU is out of scope.
