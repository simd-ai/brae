# Linear solvers. PCG, PBiCG(Stab), DILU/DIC, Gauss-Seidel smoothers (host and colour-order device sweeps), GAMG and
# brae's AMG-PCG, SpMV, reductions and the CUDA-graph loops, with their identity and memory checks.
#
# Included by tests/CMakeLists.txt.

add_executable(test_spmv ${PROJECT_SOURCE_DIR}/tests/test_spmv.cu)
target_include_directories(test_spmv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_spmv PRIVATE brae_core)
add_test(NAME spmv
         COMMAND test_spmv ${PROJECT_SOURCE_DIR}/validation/matrixDump)

add_executable(test_pcg ${PROJECT_SOURCE_DIR}/tests/test_pcg.cu)
target_include_directories(test_pcg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_pcg PRIVATE brae_core)
add_test(NAME pcg
         COMMAND test_pcg ${PROJECT_SOURCE_DIR}/validation/matrixDump)

add_executable(test_gamg ${PROJECT_SOURCE_DIR}/tests/test_gamg.cu)
target_include_directories(test_gamg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gamg PRIVATE brae_core)
add_test(NAME gamg
         COMMAND test_gamg ${PROJECT_SOURCE_DIR}/validation/matrixDumpGAMG)

add_executable(test_pbicgstab ${PROJECT_SOURCE_DIR}/tests/test_pbicgstab.cu)
target_include_directories(test_pbicgstab PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_pbicgstab PRIVATE brae_core)
add_test(NAME pbicgstab
         COMMAND test_pbicgstab ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym)

add_executable(test_vector_solve ${PROJECT_SOURCE_DIR}/tests/test_vector_solve.cu)
target_include_directories(test_vector_solve PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_vector_solve PRIVATE brae_core)
add_test(NAME vector_solve
         COMMAND test_vector_solve ${PROJECT_SOURCE_DIR}/validation/matrixDumpAsym)

add_test(NAME solver_nsweeps_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/solver_nsweeps_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/simpleBoxIO)
set_tests_properties(solver_nsweeps_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME solver_iterbounds_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/solver_iterbounds_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/simpleBoxIO)
set_tests_properties(solver_iterbounds_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME gs_smoother_notice
         COMMAND ${PROJECT_SOURCE_DIR}/tests/gs_smoother_notice.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(gs_smoother_notice PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

add_executable(gs_ladder ${PROJECT_SOURCE_DIR}/tests/gs_ladder.cu)
target_include_directories(gs_ladder PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(gs_ladder PRIVATE brae_core)

add_test(NAME gs_ladder_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/gs_ladder_vs_openfoam.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(gs_ladder_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME gs_device_loop_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/gs_device_loop_identity.sh
                 ${PROJECT_SOURCE_DIR}/validation/T3A)
set_tests_properties(gs_device_loop_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME bicg_device_loop_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/bicg_device_loop_identity.sh)
set_tests_properties(bicg_device_loop_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# The series must survive the host-loop fallback too: polyDeg was not forwarded there until 2026-09-12,
# so a solve that fell back ran the bare diagonal where the caller asked for a degree-d Neumann series.
add_test(NAME bicg_polydeg_host_loop
         COMMAND ${PROJECT_SOURCE_DIR}/tests/bicg_polydeg_host_loop.sh)
set_tests_properties(bicg_polydeg_host_loop PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME gs_fused_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/gs_fused_identity.sh)
set_tests_properties(gs_fused_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME gs_level_gather_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/gs_level_gather_identity.sh)
set_tests_properties(gs_level_gather_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME gs_host_smoother_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/gs_host_smoother_identity.sh)
set_tests_properties(gs_host_smoother_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME gradu_memo_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/gradu_memo_identity.sh)
# 161 s by itself; 635 s and then past 900 s in two full suites (2026-10-08), sharing the GPU with whatever ran
# beside it. The limit is for a hang and has to clear that.
set_tests_properties(gradu_memo_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 2400)

add_test(NAME normfactor_device_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/normfactor_device_identity.sh)
set_tests_properties(normfactor_device_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME dilu_single_block_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/dilu_single_block_identity.sh)
set_tests_properties(dilu_single_block_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

add_test(NAME pcg_device_loop_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/pcg_device_loop_identity.sh)
set_tests_properties(pcg_device_loop_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

add_test(NAME dilu_vs_openfoam
         COMMAND ${PROJECT_SOURCE_DIR}/tests/dilu_vs_openfoam.sh)
set_tests_properties(dilu_vs_openfoam PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

add_test(NAME host_smoother_memory
         COMMAND ${PROJECT_SOURCE_DIR}/tests/host_smoother_memory.sh)
set_tests_properties(host_smoother_memory PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

add_test(NAME device_memory_flat
         COMMAND ${PROJECT_SOURCE_DIR}/tests/device_memory_flat.sh)
set_tests_properties(device_memory_flat PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

add_executable(test_linear_solver_setup ${PROJECT_SOURCE_DIR}/tests/test_linear_solver_setup.cu)
target_include_directories(test_linear_solver_setup PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_linear_solver_setup PRIVATE brae_core)
add_test(NAME linear_solver_setup COMMAND test_linear_solver_setup)

# E2/E3 (dict_audit): brae must SAY when it substitutes its own linear solver / preconditioner.
add_executable(test_solver_notices ${PROJECT_SOURCE_DIR}/tests/test_solver_notices.cu)
target_include_directories(test_solver_notices PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_solver_notices PRIVATE brae_core)
add_test(NAME solver_notices COMMAND test_solver_notices)

# PBiCG with DILU on the device against the host reference (waves/mangroveInteraction's k and epsilon)
add_executable(test_device_pbicg ${PROJECT_SOURCE_DIR}/tests/test_device_pbicg.cu)
target_include_directories(test_device_pbicg PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_device_pbicg PRIVATE brae_core)
add_test(NAME device_pbicg COMMAND test_device_pbicg)
set_tests_properties(device_pbicg PROPERTIES SKIP_RETURN_CODE 77)

# brae::smoothSolver, the host reference: smoothSolver::solve's loop (what nIterations counts, nSweeps,
# minIter, the negative-sweep branch, the mesh-order refusal) against the device's OpenFOAM-held solver,
# with PBiCGStab as the control. The SWEEP is gs_ladder's, against OpenFOAM's own per-sweep residuals.
add_executable(test_smooth_solver_cpp ${PROJECT_SOURCE_DIR}/tests/test_smooth_solver_cpp.cu)
target_include_directories(test_smooth_solver_cpp PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_smooth_solver_cpp PRIVATE brae_core)
add_test(NAME smooth_solver_cpp COMMAND test_smooth_solver_cpp)

add_executable(test_dilu ${PROJECT_SOURCE_DIR}/tests/test_dilu.cu)
target_include_directories(test_dilu PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_dilu PRIVATE brae_core)
add_test(NAME dilu COMMAND test_dilu)

# AMG as the preconditioner of the ASYMMETRIC BiCGStab (OpenFOAM's `solver PBiCGStab; preconditioner GAMG`):
# the answer must not depend on the preconditioner, it must take fewer iterations, and the coarsest solve --
# a CG on a nonsymmetric operator until now -- is held to a host direct solve with that CG as the fail-proof.
add_executable(test_amg_bicgstab ${PROJECT_SOURCE_DIR}/tests/test_amg_bicgstab.cu)
target_include_directories(test_amg_bicgstab PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_amg_bicgstab PRIVATE brae_core)
add_test(NAME amg_bicgstab COMMAND test_amg_bicgstab)
add_executable(test_relaxation_final ${PROJECT_SOURCE_DIR}/tests/test_relaxation_final.cu)
target_include_directories(test_relaxation_final PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_relaxation_final PRIVATE brae_core)
add_test(NAME relaxation_final COMMAND test_relaxation_final)
add_executable(test_mean_velocity_force ${PROJECT_SOURCE_DIR}/tests/test_mean_velocity_force.cu)
target_include_directories(test_mean_velocity_force PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_mean_velocity_force PRIVATE brae_core)
add_test(NAME mean_velocity_force COMMAND test_mean_velocity_force)
add_executable(test_interface_invariants_real ${PROJECT_SOURCE_DIR}/tests/test_interface_invariants_real.cu)
target_include_directories(test_interface_invariants_real PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_interface_invariants_real PRIVATE brae_core)
add_test(NAME interface_invariants_real COMMAND test_interface_invariants_real)

# GPU offload G0: device foundation (DeviceBuffer + BLAS-1 kernels)
add_executable(test_gpu_blas ${PROJECT_SOURCE_DIR}/tests/test_gpu_blas.cu)
target_include_directories(test_gpu_blas PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_blas PRIVATE brae_core)
add_test(NAME gpu_blas COMMAND test_gpu_blas)

# deviceReadScalar's mailbox: the published value must be bit-for-bit the blocking copy's, in both modes,
# with the unordered-producer control and the bounded wait's throw.
add_executable(test_scalar_mailbox ${PROJECT_SOURCE_DIR}/tests/test_scalar_mailbox.cu)
target_include_directories(test_scalar_mailbox PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_scalar_mailbox PRIVATE brae_core)
add_test(NAME scalar_mailbox COMMAND test_scalar_mailbox)

# GPU offload G1: device lduMatrix SpMV (Amul)
add_executable(test_gpu_spmv ${PROJECT_SOURCE_DIR}/tests/test_gpu_spmv.cu)
target_include_directories(test_gpu_spmv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_spmv PRIVATE brae_core)
add_test(NAME gpu_spmv COMMAND test_gpu_spmv ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload G2: device-resident Jacobi-PCG
add_executable(test_gpu_pcg ${PROJECT_SOURCE_DIR}/tests/test_gpu_pcg.cu)
target_include_directories(test_gpu_pcg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_pcg PRIVATE brae_core)
add_test(NAME gpu_pcg COMMAND test_gpu_pcg ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload (#6): AMG preconditioner
add_executable(test_gpu_amg ${PROJECT_SOURCE_DIR}/tests/test_gpu_amg.cu)
target_include_directories(test_gpu_amg PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_amg PRIVATE brae_core)
# The AMG coarsest grid's direct solve, kind by kind (BRAE_AMG_COARSE_SOLVE): a benchmark, not a test
add_executable(bench_amg_coarse_solve ${PROJECT_SOURCE_DIR}/tests/bench_amg_coarse_solve.cu)
target_include_directories(bench_amg_coarse_solve PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(bench_amg_coarse_solve PRIVATE brae_core)
add_test(NAME gpu_amg COMMAND test_gpu_amg ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# The AMG coarsest grid's precomputed inverse against the LU and a long double oracle, the whole preconditioner's
# symmetry and definiteness, and six broken-inverse controls run as children (BRAE_CONTROL_COARSE_INVERSE)
add_executable(test_amg_coarse_inverse ${PROJECT_SOURCE_DIR}/tests/test_amg_coarse_inverse.cu)
target_include_directories(test_amg_coarse_inverse PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_amg_coarse_inverse PRIVATE brae_core)
add_test(NAME amg_coarse_inverse COMMAND test_amg_coarse_inverse ${PROJECT_SOURCE_DIR}/validation/pitzDaily)
set_tests_properties(amg_coarse_inverse PROPERTIES RESOURCE_LOCK brae_gpu LABELS fast)

# GPU offload #7: thread-block clusters + DSM, foundational cluster reduction vs the global reduction.
add_executable(test_gpu_cluster ${PROJECT_SOURCE_DIR}/tests/test_gpu_cluster.cu)
target_include_directories(test_gpu_cluster PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_cluster PRIVATE brae_core)
add_test(NAME gpu_cluster COMMAND test_gpu_cluster)

# GPU offload #7b: AMG coarse Jacobi solve fused into one cluster+DSM kernel vs the unfused loop.
add_executable(test_gpu_amg_fused ${PROJECT_SOURCE_DIR}/tests/test_gpu_amg_fused.cu)
target_include_directories(test_gpu_amg_fused PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_amg_fused PRIVATE brae_core)
add_test(NAME gpu_amg_fused COMMAND test_gpu_amg_fused ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload #7a: cluster+DSM SpMV vs the global-memory deviceAmul (bit-exact + timing).
add_executable(test_gpu_cluster_spmv ${PROJECT_SOURCE_DIR}/tests/test_gpu_cluster_spmv.cu)
target_include_directories(test_gpu_cluster_spmv PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_cluster_spmv PRIVATE brae_core)
add_test(NAME gpu_cluster_spmv COMMAND test_gpu_cluster_spmv ${PROJECT_SOURCE_DIR}/validation/pitzDaily ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurbBig)

# GPU offload #7c: CUDA-graph replay of the AMG V-cycle vs direct kernel launches (identical result + timing).
add_executable(test_gpu_amg_graph ${PROJECT_SOURCE_DIR}/tests/test_gpu_amg_graph.cu)
target_include_directories(test_gpu_amg_graph PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_gpu_amg_graph PRIVATE brae_core)
add_test(NAME gpu_amg_graph COMMAND test_gpu_amg_graph ${PROJECT_SOURCE_DIR}/validation/pitzDaily ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurbBig)

# The coarse matrices of a fine matrix handed with one off-diagonal: one coarse off-diagonal a level, the same bits
# and solves as two, and a hierarchy switched to two off-diagonals drops its captured graphs (with its control).
add_executable(test_amg_one_off_diagonal ${PROJECT_SOURCE_DIR}/tests/test_amg_one_off_diagonal.cu)
target_include_directories(test_amg_one_off_diagonal PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_amg_one_off_diagonal PRIVATE brae_core)
add_test(NAME amg_one_off_diagonal COMMAND test_amg_one_off_diagonal ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# The AMG's single-precision CSR values gathered through each grid's addressing: the stored source list's
# bits (BRAE_AMG_CSR_SRC=1 keeps the list), a smaller footprint, and a control that changes the cycle.
add_test(NAME amg_csr_rows_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/amg_csr_rows_identity.sh)
set_tests_properties(amg_csr_rows_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# The pressure AMG-PCG's double-precision rows (the PCG's product, the double cycle's finest and coarse grids) byte for byte the face
# form, in both cycle precisions; controls: the rows in another order, the values never gathered again.
add_test(NAME amg_double_rows_identity
         COMMAND ${PROJECT_SOURCE_DIR}/tests/amg_double_rows_identity.sh)
set_tests_properties(amg_double_rows_identity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

# The double cycle's fused level products (amgFusedLevelProducts, opt-in): the cycle from every grid, captured, and a whole
# AMG-PCG solve the unfused bits; control: the post-smooth's row summed from its last entry. Two arms, as the settings are read
# once a process: rows on every grid, and on the finest grid alone (the switch must then act there only).
add_executable(test_amg_fused_level_products ${PROJECT_SOURCE_DIR}/tests/test_amg_fused_level_products.cu)
target_include_directories(test_amg_fused_level_products PRIVATE ${BRAE_HEADER_DIRS})
target_link_libraries(test_amg_fused_level_products PRIVATE brae_core)
add_test(NAME amg_fused_level_products
         COMMAND test_amg_fused_level_products ${PROJECT_SOURCE_DIR}/validation/pitzDaily all)
add_test(NAME amg_fused_level_products_finest
         COMMAND test_amg_fused_level_products ${PROJECT_SOURCE_DIR}/validation/pitzDaily finest)
set_tests_properties(amg_fused_level_products amg_fused_level_products_finest PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

# The BiCGStab graph cache keyed on the field it solves holds one graph per solved field: the velocity corrector
# writes U into its own block (a new block each iteration had become a new cache entry an iteration once boxy5's
# memory step 4 moved the pool's free list). Control: BRAE_CONTROL_U_CORRECTOR_NEW_BLOCK=1, the old new block.
add_test(NAME bicg_graph_cache_bounded
         COMMAND ${PROJECT_SOURCE_DIR}/tests/bicg_graph_cache_bounded.sh)
set_tests_properties(bicg_graph_cache_bounded PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# GPU offload #7c-loop: cached V-cycle graph replayed across SIMPLE steps (persistent buffers) vs direct, in-loop A/B.
add_executable(test_gpu_loop_graph ${PROJECT_SOURCE_DIR}/tests/test_gpu_loop_graph.cu)
target_include_directories(test_gpu_loop_graph PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_loop_graph PRIVATE brae_core)
add_test(NAME gpu_loop_graph COMMAND test_gpu_loop_graph ${PROJECT_SOURCE_DIR}/validation/pitzDaily)

# GPU offload #8 re-measure: GPU-vs-CPU crossover with all #6/#7 wins (warm + min-of-repeats controlled timing).
add_executable(test_gpu_crossover ${PROJECT_SOURCE_DIR}/tests/test_gpu_crossover.cu)
target_include_directories(test_gpu_crossover PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/src/applications/solvers/simpleFoam)
target_link_libraries(test_gpu_crossover PRIVATE brae_core)
add_test(NAME gpu_crossover COMMAND test_gpu_crossover ${PROJECT_SOURCE_DIR}/validation/pitzDaily ${PROJECT_SOURCE_DIR}/validation/pitzDailyTurbBig)

add_executable(test_colour_gs_fused ${PROJECT_SOURCE_DIR}/tests/test_colour_gs_fused.cu)
target_include_directories(test_colour_gs_fused PRIVATE ${BRAE_HEADER_DIRS} ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_colour_gs_fused PRIVATE brae_core)
add_test(NAME colour_gs_fused COMMAND test_colour_gs_fused)
